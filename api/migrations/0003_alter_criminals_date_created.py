# Generated by Django 4.2.4 on 2023-09-07 16:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0002_alter_criminals_date_created'),
    ]

    operations = [
        migrations.AlterField(
            model_name='criminals',
            name='date_created',
            field=models.DateTimeField(auto_created=True, default='07/09/2023 16:17', max_length=20),
        ),
    ]
