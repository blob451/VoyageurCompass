"""
Management command to generate explanations for analysis results.
Usage: python manage.py generate_explanations [options]
"""

from datetime import datetime

from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from Analytics.services.explanation_service import get_explanation_service
from Data.models import AnalyticsResults

User = get_user_model()


class Command(BaseCommand):
    help = "Generate explanations for analysis results using local LLaMA model"

    def add_arguments(self, parser):
        # Symbol filter
        parser.add_argument("--symbol", type=str, help="Generate explanations for specific stock symbol only")

        # User filter
        parser.add_argument("--user", type=str, help="Generate explanations for specific user (username or email)")

        # Date range
        parser.add_argument("--since", type=str, help="Generate explanations for analyses since date (YYYY-MM-DD)")

        parser.add_argument("--until", type=str, help="Generate explanations for analyses until date (YYYY-MM-DD)")

        # Detail level
        parser.add_argument(
            "--detail-level",
            type=str,
            choices=["summary", "standard", "detailed"],
            default="standard",
            help="Explanation detail level (default: standard)",
        )

        # Batch size
        parser.add_argument(
            "--batch-size", type=int, default=10, help="Number of analyses to process at once (default: 10)"
        )

        # Force regeneration
        parser.add_argument("--force", action="store_true", help="Regenerate explanations even if they already exist")

        # Dry run
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be processed without actually generating explanations",
        )

        # Limit
        parser.add_argument("--limit", type=int, help="Maximum number of analyses to process")

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS("Starting explanation generation process..."))

        # Initialize explanation service
        explanation_service = get_explanation_service()

        if not explanation_service.is_enabled():
            raise CommandError("Explanation service is not enabled or available")

        # Get service status
        status = explanation_service.get_service_status()
        self.stdout.write(f"LLM Service Status: {status}")

        if not status.get("llm_available", False):
            self.stdout.write(self.style.WARNING("LLM service not available - will use template fallback"))

        # Build query filters
        queryset = AnalyticsResults.objects.select_related("stock", "user")

        # Apply filters
        if options["symbol"]:
            symbol = options["symbol"].upper()
            queryset = queryset.filter(stock__symbol=symbol)
            self.stdout.write(f"Filtering by symbol: {symbol}")

        if options["user"]:
            try:
                from django.db import models

                user = User.objects.get(models.Q(username=options["user"]) | models.Q(email=options["user"]))
                queryset = queryset.filter(user=user)
                self.stdout.write(f"Filtering by user: {user.username}")
            except User.DoesNotExist:
                raise CommandError(f"User '{options['user']}' not found")

        if options["since"]:
            try:
                since_date = datetime.strptime(options["since"], "%Y-%m-%d").date()
                queryset = queryset.filter(as_of__date__gte=since_date)
                self.stdout.write(f"Filtering since: {since_date}")
            except ValueError:
                raise CommandError("Invalid date format for --since. Use YYYY-MM-DD")

        if options["until"]:
            try:
                until_date = datetime.strptime(options["until"], "%Y-%m-%d").date()
                queryset = queryset.filter(as_of__date__lte=until_date)
                self.stdout.write(f"Filtering until: {until_date}")
            except ValueError:
                raise CommandError("Invalid date format for --until. Use YYYY-MM-DD")

        # Filter out analyses that already have explanations (unless force is used)
        if not options["force"]:
            queryset = queryset.filter(narrative_text__isnull=True)
            self.stdout.write("Excluding analyses that already have explanations (use --force to override)")

        # Apply limit
        if options["limit"]:
            queryset = queryset[: options["limit"]]

        # Order by most recent first
        queryset = queryset.order_by("-as_of")

        total_count = queryset.count()
        self.stdout.write(f"Found {total_count} analyses to process")

        if total_count == 0:
            self.stdout.write(self.style.WARNING("No analyses found matching criteria"))
            return

        if options["dry_run"]:
            self.stdout.write(self.style.WARNING("DRY RUN - No explanations will be generated"))
            for analysis in queryset[:10]:  # Show first 10
                self.stdout.write(f"  - {analysis.stock.symbol} ({analysis.user.username}) - {analysis.as_of}")
            if total_count > 10:
                self.stdout.write(f"  ... and {total_count - 10} more")
            return

        # Process in batches
        batch_size = options["batch_size"]
        detail_level = options["detail_level"]
        processed = 0
        successful = 0
        failed = 0

        self.stdout.write(f"Processing {total_count} analyses in batches of {batch_size}")
        self.stdout.write(f"Detail level: {detail_level}")

        for i in range(0, total_count, batch_size):
            batch = list(queryset[i : i + batch_size])

            self.stdout.write(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} analyses)...")

            for analysis in batch:
                try:
                    self.stdout.write(
                        f"  Generating explanation for {analysis.stock.symbol} "
                        f"(ID: {analysis.id}, User: {analysis.user.username})"
                    )

                    # Generate explanation
                    explanation_result = explanation_service.explain_prediction_single(
                        analysis, detail_level=detail_level, user=analysis.user
                    )

                    if explanation_result:
                        # Update the database
                        analysis.explanations_json = {
                            "indicators_explained": explanation_result.get("indicators_explained", []),
                            "risk_factors": explanation_result.get("risk_factors", []),
                            "recommendation": explanation_result.get("recommendation", "HOLD"),
                        }
                        analysis.explanation_method = explanation_result.get("method", "unknown")
                        analysis.explanation_version = "1.0"
                        analysis.narrative_text = explanation_result.get("content", "")
                        analysis.explanation_confidence = explanation_result.get("confidence_score", 0.0)
                        analysis.explained_at = timezone.now()
                        analysis.save()

                        successful += 1
                        self.stdout.write(
                            self.style.SUCCESS(
                                f"    [SUCCESS] Generated ({explanation_result.get('word_count', 0)} words, "
                                f"{explanation_result.get('generation_time', 0):.2f}s, "
                                f"method: {explanation_result.get('method', 'unknown')})"
                            )
                        )
                    else:
                        failed += 1
                        self.stdout.write(self.style.ERROR(f"    [FAILED] Failed to generate explanation"))

                except Exception as e:
                    failed += 1
                    self.stdout.write(self.style.ERROR(f"    [ERROR] Error: {str(e)}"))

                processed += 1

            # Progress update
            progress = (processed / total_count) * 100
            self.stdout.write(
                f"Progress: {processed}/{total_count} ({progress:.1f}%) - " f"Success: {successful}, Failed: {failed}"
            )

        # Final summary
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write(self.style.SUCCESS("Explanation generation completed!"))
        self.stdout.write(f"Total processed: {processed}")
        self.stdout.write(self.style.SUCCESS(f"Successful: {successful}"))
        if failed > 0:
            self.stdout.write(self.style.ERROR(f"Failed: {failed}"))

        success_rate = (successful / processed * 100) if processed > 0 else 0
        self.stdout.write(f"Success rate: {success_rate:.1f}%")

        if failed > 0:
            self.stdout.write(self.style.WARNING("Some explanations failed to generate. Check logs for details."))
