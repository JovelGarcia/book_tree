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
        ('p',   'Pending'),
        ('pr',  'Processing'),
        ('c',   'Completed'),
        ('f',   'Failed'),
        ('ns',  'Needs Scope'),
        ('nc',  'No Category'),
        ('dup', 'Duplicate'),   # cache hit — refer to cached request
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
    wiki_slug = models.CharField(max_length=255, blank=True)
    wiki_url  = models.URLField(blank=True)

    # Presentational metadata (best-effort, populated by fetch_media_metadata)
    release_year = models.CharField(max_length=4,   blank=True, default='')
    creators     = models.JSONField(default=list,   blank=True)
    genres       = models.JSONField(default=list,   blank=True)

    # Strategy / category reasoning
    resolution_strategy = models.CharField(
        max_length=20,
        choices=RESOLUTION_STRATEGY,
        blank=True,
        default='',
    )
    chosen_category     = models.CharField(max_length=500, blank=True, default='')
    metadata_categories = models.JSONField(default=list,   blank=True)
    strategy_reasoning  = models.TextField(blank=True, default='')

    status        = models.CharField(max_length=3, choices=STATUS_CHOICES, default='p')
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
    media  = models.ForeignKey('MediaRequest', on_delete=models.CASCADE,
                               related_name='relationships')
    source = models.ForeignKey('Character', on_delete=models.CASCADE,
                               related_name='outgoing_rels', default='')
    target = models.ForeignKey('Character', on_delete=models.CASCADE,
                               related_name='incoming_rels', default='')
    relationship_type = models.CharField(max_length=50)
    description       = models.TextField(blank=True, default='')

    class Meta:
        unique_together = ('media', 'source', 'target')

    def __str__(self):
        return f"{self.source.name} → {self.target.name} ({self.relationship_type})"