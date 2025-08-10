# Generated data migration to populate portfolio user field

import secrets
import string
from django.db import migrations

def populate_portfolio_users(apps, schema_editor):
    """
    Assign existing portfolios to the first available user.
    In production, you should implement proper user assignment logic.
    """
    Portfolio = apps.get_model('Data', 'Portfolio')
    User = apps.get_model('auth', 'User')
    
    # Get portfolios without users
    portfolios_without_users = Portfolio.objects.filter(user__isnull=True)
    
    if portfolios_without_users.exists():
        # Try to get the first superuser, then any user
        try:
            default_user = User.objects.filter(is_superuser=True).first()
            if not default_user:
                default_user = User.objects.first()
            
            if default_user:
                updated_count = portfolios_without_users.update(user=default_user)
                print(f"Assigned {updated_count} portfolios to user {default_user.username}")
            else:
                # Create a default admin user if no users exist
                # Generate a secure random password
                alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
                secure_password = ''.join(secrets.choice(alphabet) for i in range(16))
                
                default_user = User.objects.create_user(
                    username='admin',
                    email='admin@example.com',
                    password=secure_password,
                    is_superuser=True,
                    is_staff=True
                )
                updated_count = portfolios_without_users.update(user=default_user)
                print(f"Created default admin user and assigned {updated_count} portfolios")
                # SECURITY: Password not logged to console for security reasons
                # Use Django's createsuperuser command or reset password via email in production
                
        except Exception as e:
            print(f"Error populating portfolio users: {e}")
            raise

def reverse_populate_portfolio_users(apps, schema_editor):
    """Reverse migration - set user field to null"""
    Portfolio = apps.get_model('Data', 'Portfolio')
    Portfolio.objects.all().update(user=None)

class Migration(migrations.Migration):

    dependencies = [
        ('Data', '0003_portfolio_user'),
    ]

    operations = [
        migrations.RunPython(
            populate_portfolio_users,
            reverse_populate_portfolio_users,
        ),
    ]