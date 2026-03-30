from django.db import models
from django.utils import timezone


class MediaRequest(models.Model):
    MEDIA_TYPE_CHOICES = [
        ('anime', 'Anime'),
        ('tv',    'TV Show'),
        ('movie', 'Movie'),
        ('game',  'Game'),
        ('book',  'Book'),
    ]
    STATUS_CHOICES = [
        ('p',  'Pending'),
        ('pr', 'Processing'),
        ('c',  'Completed'),
        ('f',  'Failed'),
        ('ns', 'Needs Scope'),       # dedicated wiki exists, but no title-scoped category
        ('nc', 'No Category'),       # best wiki found, but zero character categories
    ]
    RESOLUTION_STRATEGY = [
        ('dedicated_scoped',   'Dedicated wiki, scoped category'),
        ('umbrella_scoped',    'Umbrella wiki, scoped category'),
        ('dedicated_unscoped', 'Dedicated wiki, unscoped category'),
        ('no_category',        'No useful character category'),
    ]

    title      = models.CharField(max_length=255)
    media_type = models.CharField(max_length=10, choices=MEDIA_TYPE_CHOICES)

    # Resolved once the task starts
    wiki_slug  = models.CharField(max_length=255, blank=True)
    wiki_url   = models.URLField(blank=True)

    # Strategy / category reasoning
    resolution_strategy = models.CharField(
        max_length=20,
        choices=RESOLUTION_STRATEGY,
        blank=True,
        default='',
    )
    chosen_category      = models.CharField(max_length=500, blank=True, default='')
    metadata_categories  = models.JSONField(default=list, blank=True)
    strategy_reasoning   = models.TextField(blank=True, default='')

    status        = models.CharField(max_length=2, choices=STATUS_CHOICES, default='p')
    error_message = models.TextField(blank=True)
    submitted_at  = models.DateTimeField(default=timezone.now)
    completed_at  = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-submitted_at']

    def __str__(self):
        return f"{self.title} ({self.get_media_type_display()}) [{self.get_status_display()}]"


class Character(models.Model):
    media         = models.ForeignKey(MediaRequest, on_delete=models.CASCADE, related_name='characters')
    name          = models.CharField(max_length=200)
    aliases       = models.JSONField(default=list)
    description   = models.TextField(blank=True)
    image_url     = models.URLField(blank=True)
    wiki_page     = models.URLField(blank=True)
    mention_count = models.IntegerField(default=0)

    class Meta:
        ordering        = ['-mention_count']
        unique_together = ['media', 'name']

    def __str__(self):
        return self.name


class Relationship(models.Model):
    media        = models.ForeignKey(MediaRequest, on_delete=models.CASCADE, related_name='relationships')
    character_1  = models.ForeignKey(Character, on_delete=models.CASCADE, related_name='relationships_as_character_1')
    character_2  = models.ForeignKey(Character, on_delete=models.CASCADE, related_name='relationships_as_character_2')

    relationship_type    = models.CharField(max_length=100)
    relationship_subtype = models.CharField(max_length=100, blank=True)
    confidence           = models.FloatField(default=0.0)
    evidence             = models.JSONField(default=list, blank=True)

    class Meta:
        unique_together = ['media', 'character_1', 'character_2', 'relationship_type']

    def __str__(self):
        return f"{self.character_1.name} —[{self.relationship_type}]→ {self.character_2.name}"