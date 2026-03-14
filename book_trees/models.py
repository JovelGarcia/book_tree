from django.db import models
from django.utils import timezone

# Create your models here.

class EpubFile(models.Model):
    file = models.FileField(upload_to='epubs/%Y/%m/%d')

    original_filename = models.CharField(max_length=255)

    uploaded_at = models.DateTimeField(default=timezone.now)
    processed = models.BooleanField(default=False)

    STATUS_CHOICES = [
        ('p', 'pending'),
        ('pr', 'processing'),
        ('c', 'completed'),
        ('f', 'failed'),
    ]
    status = models.CharField(max_length=2, choices=STATUS_CHOICES, default='p')

    class Meta:
        ordering = ['uploaded_at']

    def __str__(self):
        return f"{self.original_filename} - {self.uploaded_at.strftime('%Y-%m-%d')}"

class Section(models.Model):
    epub = models.ForeignKey(
        EpubFile,
        on_delete=models.CASCADE,
        related_name='sections'
    )
    title = models.CharField(max_length=500)
    order = models.IntegerField()

    class Meta:
        ordering = ['epub', 'order']
        unique_together = ['epub', 'order']

    def __str__(self):
        return f"{self.epub.original_filename} - {self.title}"


class Character(models.Model):
    epub = models.ForeignKey(EpubFile, on_delete=models.CASCADE, related_name='characters')
    name = models.CharField(max_length=200)
    syntactic_score = models.IntegerField(default=0)
    aliases = models.JSONField(default=list)
    mention_count = models.IntegerField(default=0)
    first_appearance_chapter = models.IntegerField(null=True)

    class Meta:
        ordering = ['-mention_count']
        unique_together = ['epub', 'name']

    def __str__(self):
        return f"{self.name}"

class Chapter(models.Model):
    epub = models.ForeignKey(
        EpubFile,
        on_delete=models.CASCADE,
        related_name='chapters'
    )
    section = models.ForeignKey(
        Section,
        on_delete=models.CASCADE,
        related_name='chapters',
        null=True,
        blank=True
    )
    title = models.CharField(max_length=500, blank=True)
    content = models.TextField()
    chapter_number = models.IntegerField()
    annotated_sentences = models.JSONField(default=list, blank=True)

    narrators = models.ManyToManyField(
        Character,
        related_name='narrated_chapters',
        blank=True
    )

    class Meta:
        ordering = ['epub', 'section__order', 'chapter_number']
        unique_together = ['epub', 'section', 'chapter_number']

    def __str__(self):
        if self.section:
            return f"{self.section.title} - Chapter {self.chapter_number}"
        return f"{self.epub.original_filename} - Chapter {self.chapter_number}"

class LabeledSentence(models.Model):
    epub = models.ForeignKey(EpubFile, on_delete=models.CASCADE, related_name='labeled_sentences', null=True, blank=True)
    text = models.TextField()
    entities = models.JSONField(
        default=list,
        help_text="""
        spaCy-compatible span format:
        [{"start": 0, "end": 5, "label": "PERSON"}, ...]
        """
    )
    source = models.CharField(max_length=100, blank=True, help_text="Book title or origin")
    labeled_by = models.CharField(max_length=100, blank=True)
    labeled_at = models.DateTimeField(auto_now_add=True)
    is_reviewed = models.BooleanField(default=False)

    def __str__(self):
        return self.text[:60]

class Relationship(models.Model):
    epub = models.ForeignKey(EpubFile, on_delete=models.CASCADE, related_name='relationships')
    character_1 = models.ForeignKey(Character, on_delete=models.CASCADE, related_name='relationships_as_character_1')
    character_2 = models.ForeignKey(Character, on_delete=models.CASCADE, related_name='relationships_as_character_2')
    relationship_type = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    # More detailed subtype returned by the LLM
    relationship_subtype = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        help_text="More specific relationship label (brother, captor, rival, etc)"
    )

    evidence = models.JSONField(
        default=list,
        blank=True,
        help_text="""
        List of evidence objects:
        [
            {
                "chapter": int,
                "specific_type": str,
                "confidence": float,
                "chunks": [
                    {
                        "context": str,
                        "characters_in_context": [str],
                        "sentence_index": int
                    }
                ]
            }
        ]
        """
    )


    class Meta:
        unique_together = ['epub', 'character_1', 'character_2', 'relationship_type']

    def __str__(self):
        return f"{self.character_1.name} - {self.relationship_type} - {self.character_2.name}"
