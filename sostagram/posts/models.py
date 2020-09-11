from django.db import models
from django.contrib.auth.models import User
# Create your models here.

class Post(models.class (models.Model):

    user = models.models.ForeignKey(User, on_delete=models.CASCADE)
    body = models.TextField()
    created_at = models.DatetimeField()

    def __str__(self):
        return f'{self.user.get_username()} : {self.body}
