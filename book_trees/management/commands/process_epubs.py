"""
Management command to process EPUB files with various processing options

Usage:
    # Full processing pipeline (chapters + characters + validation + relationships)
    python manage.py process_epubs --full --api-key YOUR_API_KEY

    # Process specific EPUB with full pipeline
    python manage.py process_epubs --epub-id 5 --full --api-key YOUR_API_KEY

    # Individual processing steps
    python manage.py process_epubs --chapters-only
    python manage.py process_epubs --characters-only
    python manage.py process_epubs --validate-characters --api-key YOUR_API_KEY
    python manage.py process_epubs --relationships-only --api-key YOUR_API_KEY

    # Reprocess all EPUBs
    python manage.py process_epubs --reprocess --full --api-key YOUR_API_KEY
"""

from django.core.management.base import BaseCommand
from django.conf import settings
from book_trees.models import EpubFile
from book_trees.processing import (
    process_epub_file,
    extract_characters_with_chunks,
    validate_and_deduplicate_characters_with_llm,
    extract_relationships_with_llm,
    process_book_complete
)


class Command(BaseCommand):
    help = 'Process EPUB files with various processing steps'

    def add_arguments(self, parser):
        # Target selection
        parser.add_argument(
            '--epub-id',
            type=int,
            help='Process a specific EPUB by ID',
        )

        # Processing modes
        parser.add_argument(
            '--full',
            action='store_true',
            help='Run full processing pipeline (chapters + characters + validation + relationships)',
        )
        parser.add_argument(
            '--chapters-only',
            action='store_true',
            help='Extract chapters only',
        )
        parser.add_argument(
            '--characters-only',
            action='store_true',
            help='Extract characters with NER (requires chapters to exist)',
        )
        parser.add_argument(
            '--validate-characters',
            action='store_true',
            help='Validate and deduplicate characters with LLM (requires API key)',
        )
        parser.add_argument(
            '--relationships-only',
            action='store_true',
            help='Extract relationships with LLM (requires API key and characters)',
        )

        # Options
        parser.add_argument(
            '--reprocess',
            action='store_true',
            help='Reprocess EPUBs even if already completed',
        )
        parser.add_argument(
            '--api-key',
            type=str,
            help='Google API key for LLM-based processing',
        )

    def handle(self, *args, **options):
        epub_id = options.get('epub_id')
        reprocess = options.get('reprocess')
        api_key = options.get('api_key') or getattr(settings, 'GOOGLE_API_KEY', None)

        # Validate API key for LLM-based operations
        llm_operations = [
            options.get('validate_characters'),
            options.get('relationships_only'),
            options.get('full')
        ]
        if any(llm_operations) and not api_key:
            self.stdout.write(self.style.ERROR(
                "Error: API key required for LLM-based processing. "
                "Provide via --api-key or set GOOGLE_API_KEY in settings."
            ))
            return

        # Determine which EPUBs to process
        if epub_id:
            try:
                epubs = [EpubFile.objects.get(id=epub_id)]
                self.stdout.write(f"Processing EPUB: {epubs[0].original_filename}\n")
            except EpubFile.DoesNotExist:
                self.stdout.write(self.style.ERROR(f"EPUB with ID {epub_id} not found"))
                return
        else:
            if reprocess:
                epubs = EpubFile.objects.all()
                self.stdout.write(f"Processing all {epubs.count()} EPUBs...\n")
            else:
                epubs = EpubFile.objects.filter(status='p')
                self.stdout.write(f"Processing {epubs.count()} pending EPUBs...\n")

        # Process each EPUB
        success_count = 0
        fail_count = 0

        for epub in epubs:
            self.stdout.write(f"\n{'='*60}")
            self.stdout.write(f"Processing: {epub.original_filename}")
            self.stdout.write(f"{'='*60}")

            try:
                # Clean up for reprocessing
                if reprocess:
                    epub.relationships.all().delete()
                    epub.characters.all().delete()
                    epub.chapters.all().delete()
                    epub.status = 'p'
                    epub.save()

                # Full processing pipeline
                if options.get('full'):
                    stats = process_book_complete(epub.id, api_key)
                    self.stdout.write(self.style.SUCCESS("\n✓ Full processing complete!"))
                    self.stdout.write(f"  Chapters: {stats['chapters']}")
                    self.stdout.write(f"  Characters: {stats['original_characters']} → {stats['final_characters']} "
                                    f"({stats['character_reduction']})")
                    self.stdout.write(f"  Invalid removed: {stats['invalid_removed']}")
                    self.stdout.write(f"  Groups merged: {stats['groups_merged']}")
                    self.stdout.write(f"  Relationships: {stats['relationships']}")
                    success_count += 1

                # Individual processing steps
                else:
                    step_count = 0

                    if options.get('chapters_only'):
                        self.stdout.write("Extracting chapters...")
                        success = process_epub_file(epub.id)
                        if success:
                            chapter_count = epub.chapters.count()
                            self.stdout.write(self.style.SUCCESS(f"✓ Extracted {chapter_count} chapters"))
                            step_count += 1
                        else:
                            raise Exception("Chapter extraction failed")

                    if options.get('characters_only'):
                        if not epub.chapters.exists():
                            self.stdout.write(self.style.WARNING(
                                "⚠ No chapters found. Run --chapters-only first."
                            ))
                        else:
                            self.stdout.write("Extracting characters with NER...")
                            char_count = extract_characters_with_chunks(epub.id)
                            self.stdout.write(self.style.SUCCESS(
                                f"✓ Found {char_count} potential character names"
                            ))
                            step_count += 1

                    if options.get('validate_characters'):
                        if not epub.characters.exists():
                            self.stdout.write(self.style.WARNING(
                                "⚠ No characters found. Run --characters-only first."
                            ))
                        else:
                            self.stdout.write("Validating and deduplicating characters with LLM...")
                            stats = validate_and_deduplicate_characters_with_llm(epub.id, api_key)
                            self.stdout.write(self.style.SUCCESS(
                                f"✓ Characters: {stats['original_count']} → {stats['final_count']} "
                                f"({stats['reduction']})"
                            ))
                            self.stdout.write(f"  Invalid removed: {stats.get('invalid_count', 0)}")
                            self.stdout.write(f"  Groups merged: {stats.get('merged_count', 0)}")
                            step_count += 1

                    if options.get('relationships_only'):
                        if not epub.characters.exists():
                            self.stdout.write(self.style.WARNING(
                                "⚠ No characters found. Run character extraction first."
                            ))
                        else:
                            self.stdout.write("Extracting relationships with LLM...")
                            rel_count = extract_relationships_with_llm(epub.id, api_key)
                            self.stdout.write(self.style.SUCCESS(
                                f"✓ Found {rel_count} relationships"
                            ))
                            step_count += 1

                    if step_count > 0:
                        success_count += 1
                    elif step_count == 0:
                        self.stdout.write(self.style.WARNING(
                            "No processing steps specified. Use --full or specific step flags."
                        ))

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"✗ Failed: {str(e)}"))
                fail_count += 1

                # Update EPUB status on failure
                if epub.status != 'f':
                    epub.status = 'f'
                    epub.error_message = str(e)
                    epub.save()

        # Summary
        self.stdout.write(f"\n{'='*60}")
        self.stdout.write(self.style.SUCCESS(
            f"Completed: {success_count} successful, {fail_count} failed"
        ))
        self.stdout.write(f"{'='*60}\n")