"""
Admin configuration for Core app models.
"""

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import User

from Core.models import PasswordResetRequest, UserProfile, UserSecurityProfile


class UserSecurityProfileInline(admin.StackedInline):
    """Inline admin for UserSecurityProfile."""

    model = UserSecurityProfile
    can_delete = False
    verbose_name_plural = "Security Profile"
    fields = ("secret_question", "failed_reset_attempts", "last_reset_attempt")
    readonly_fields = ("failed_reset_attempts", "last_reset_attempt")


class UserProfileInline(admin.StackedInline):
    """Inline admin for UserProfile."""

    model = UserProfile
    can_delete = False
    verbose_name_plural = "User Profile"
    fields = ("credits", "created_at", "updated_at")
    readonly_fields = ("created_at", "updated_at")


class UserAdmin(BaseUserAdmin):
    """Extended User admin with security and user profiles."""

    inlines = (UserSecurityProfileInline, UserProfileInline)


@admin.register(PasswordResetRequest)
class PasswordResetRequestAdmin(admin.ModelAdmin):
    """Admin for password reset requests."""

    list_display = ("user", "status", "created_at", "updated_at")
    list_filter = ("status", "created_at")
    search_fields = ("user__username", "user__email", "reason")
    readonly_fields = ("user", "reason", "created_at")

    fieldsets = (
        (None, {"fields": ("user", "reason", "status")}),
        ("Admin Response", {"fields": ("admin_notes",)}),
        ("Timestamps", {"fields": ("created_at", "updated_at"), "classes": ("collapse",)}),
    )

    def save_model(self, request, obj, form, change):
        """Add logging when admin updates a reset request."""
        if change and "status" in form.changed_data:
            import logging

            logger = logging.getLogger("VoyageurCompass.admin")
            logger.info(
                f"Admin {request.user.username} changed reset request for "
                f"{obj.user.username} from {form.initial['status']} to {obj.status}"
            )

            # If approved, generate a reset token
            if obj.status == "approved":
                try:
                    security_profile = obj.user.security_profile
                    token = security_profile.generate_reset_token()
                    obj.admin_notes = f"{obj.admin_notes}\n\nReset token generated: {token}"
                except Exception:
                    obj.admin_notes = f"{obj.admin_notes}\n\nError: Could not generate reset token"

        super().save_model(request, obj, form, change)


# Re-register User admin
admin.site.unregister(User)
admin.site.register(User, UserAdmin)
