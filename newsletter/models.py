from django.db import models
# Create your models here.

class News(models.Model) :
    link = models.TextField()
    title = models.TextField()
    date = models.TextField()
    content = models.TextField()
    tag = models.TextField()
    big_category = models.TextField()
    category = models.TextField()
    team_category = models.TextField()


class New_News(models.Model) :
    link = models.TextField()
    tag = models.TextField()
    content = models.TextField()
    title = models.TextField()
    date = models.TextField()
    big_category = models.TextField()