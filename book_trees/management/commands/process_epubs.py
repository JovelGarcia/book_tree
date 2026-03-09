"""
Management command to process EPUB files with various processing options

Usage:
    # Full processing pipeline (chapters + characters + validation + relationships)
    python3 manage.py process_epubs --full --api-key YOUR_API_KEY

    # Process specific EPUB with full pipeline
    python3 manage.py process_epubs --epub-id 5 --full --api-key YOUR_API_KEY

    # Individual processing steps
    python3 manage.py process_epubs --chapters-only
    python3 manage.py process_epubs --characters-only
    python3 manage.py process_epubs --relationships-only --api-key YOUR_API_KEY

    # Continue from a partial run (e.g. after --chapters-only or --characters-only)
    python3 manage.py process_epubs --epub-id 5 --continue --api-key YOUR_API_KEY

    # Reprocess all EPUBs
    python3 manage.py process_epubs --reprocess --full --api-key YOUR_API_KEY

    # Dry run (process but do not save anything to the database)
    python3 manage.py process_epubs --full --no-save --api-key YOUR_API_KEY
"""

import zipfile
from pathlib import Path

from book_trees.post_processing import consolidate_relationships
from django.core.management.base import BaseCommand
from django.conf import settings
from django.db import transaction
from book_trees.models import EpubFile, Character, Chapter, Relationship, Section
from book_trees.processing import (
    process_epub_file,
    extract_characters_with_chunks,
    extract_relationships_with_llm,
    process_book_complete
)


class _RollbackDryRun(Exception):
    """Sentinel exception used to roll back a dry-run transaction."""


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
            '--relationships-only',
            action='store_true',
            help='Extract relationships with LLM (requires API key and characters)',
        )
        parser.add_argument(
            '--post-process',
            action='store_true',
            help="Combine relationship types"
        )
        parser.add_argument(
            '--view-raw',
            action='store_true',
            help='View raw EPUB contents (zip listing + raw XHTML) in terminal',
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
        parser.add_argument(
            '--no-save',
            action='store_true',
            help='Run processing but do not save results to the database (dry run)',
        )
        parser.add_argument(
            '--continue',
            action='store_true',
            dest='continue_processing',
            help=(
                'Resume a partial run to completion. Detects which steps are already done '
                '(chapters, characters, relationships) and runs only the remaining ones. '
                'Requires --api-key because it will proceed through LLM steps if needed.'
            ),
        )

    def handle(self, *args, **options):
        no_save = options.get('no_save')

        if no_save:
            self.stdout.write(self.style.WARNING(
                "⚠ Dry-run mode: all processing will run inside a transaction "
                "that is rolled back at the end — nothing will be saved.\n"
            ))
            try:
                with transaction.atomic():
                    self._handle(*args, **options)
                    raise _RollbackDryRun
            except _RollbackDryRun:
                self.stdout.write(self.style.WARNING(
                    "\n⚠ Dry-run complete: transaction rolled back, database unchanged."
                ))
        else:
            self._handle(*args, **options)

    def _handle(self, *args, **options):
        epub_id = options.get('epub_id')
        reprocess = options.get('reprocess')
        api_key = options.get('api_key') or getattr(settings, 'GOOGLE_API_KEY', None)

        # Validate API key for LLM-based operations
        llm_operations = [
            options.get('validate_characters'),
            options.get('relationships_only'),
            options.get('full'),
            options.get('continue_processing'),
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

        # View raw EPUB contents
        if options.get('view_raw'):
            if not epub_id:
                self.stdout.write(self.style.ERROR(
                    "--view-raw requires --epub-id"
                ))
                return

            epub = epubs[0]
            epub_path = Path(epub.file.path)

            self.stdout.write(self.style.SUCCESS(
                f"\n📘 Viewing raw EPUB: {epub.original_filename}"
            ))
            self.stdout.write(f"Path: {epub_path}\n")

            if not epub_path.exists():
                self.stdout.write(self.style.ERROR("EPUB file not found on disk"))
                return

            with zipfile.ZipFile(epub_path, 'r') as zf:
                self.stdout.write("📂 EPUB ZIP contents:\n")
                for name in zf.namelist():
                    self.stdout.write(f"  - {name}")

                self.stdout.write("\n📄 XHTML / HTML files:\n")

                for name in zf.namelist():
                    if name.endswith(('.xhtml', '.html')):
                        self.stdout.write(self.style.WARNING(f"\n--- {name} ---\n"))
                        raw = zf.read(name).decode('utf-8', errors='ignore')

                        # Prevent terminal overload
                        max_chars = 4000
                        if len(raw) > max_chars:
                            raw = raw[:max_chars] + "\n\n[...truncated...]"

                        self.stdout.write(raw)

            return  # IMPORTANT: stop further processing

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
                    Relationship.objects.filter(epub=epub).delete()
                    Character.objects.filter(epub=epub).delete()
                    Chapter.objects.filter(epub=epub).delete()
                    Section.objects.filter(epub=epub).delete()
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

                # Continue from a partial run
                elif options.get('continue_processing'):
                    self._continue_processing(epub, api_key)
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
                        process_epub_file(epub.id)
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

                    if options.get('post_process'):
                        if not epub.relationships.exists():
                            self.stdout.write(self.style.WARNING(
                                "⚠ No relationships found. Run --characters-only first."
                            ))
                        self.stdout.write("Processing relationships...")
                        success = consolidate_relationships(epub.id)
                        if success:
                            relationship_count = epub.relationships.count()
                            self.stdout.write(self.style.SUCCESS(f"✓ Consolidated to {relationship_count} relationships"))
                            step_count += 1
                        else:
                            raise Exception("Relationship consolidation failed")

                    if step_count > 0:
                        success_count += 1
                    elif step_count == 0:
                        self.stdout.write(self.style.WARNING(
                            "No processing steps specified. Use --full, --continue, or specific step flags."
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

    def _continue_processing(self, epub, api_key):
        """
        Resume a partial pipeline run to completion.

        Checks what data already exists for the epub and skips those steps,
        running only what remains in pipeline order:
            1. chapters  →  2. characters  →  3. relationships  →  4. post-process
        """
        has_chapters = epub.chapters.exists()
        has_characters = epub.characters.exists()
        has_relationships = epub.relationships.exists()

        self.stdout.write(
            f"  Resuming from: "
            f"chapters={'✓' if has_chapters else '✗'}  "
            f"characters={'✓' if has_characters else '✗'}  "
            f"relationships={'✓' if has_relationships else '✗'}"
        )

        # Step 1 — chapters
        if not has_chapters:
            self.stdout.write("  → Extracting chapters...")
            success = process_epub_file(epub.id)
            if not success:
                raise Exception("Chapter extraction failed")
            chapter_count = epub.chapters.count()
            self.stdout.write(self.style.SUCCESS(f"  ✓ Extracted {chapter_count} chapters"))
        else:
            self.stdout.write("  ✓ Chapters already present, skipping.")

        # Step 2 — characters
        if not has_characters:
            self.stdout.write("  → Extracting characters with NER...")
            char_count = extract_characters_with_chunks(epub.id)
            self.stdout.write(self.style.SUCCESS(f"  ✓ Found {char_count} potential character names"))
        else:
            self.stdout.write("  ✓ Characters already present, skipping.")

        # Step 3 — relationships
        if not has_relationships:
            self.stdout.write("  → Extracting relationships with LLM...")
            rel_count = extract_relationships_with_llm(epub.id, api_key)
            self.stdout.write(self.style.SUCCESS(f"  ✓ Found {rel_count} relationships"))
        else:
            self.stdout.write("  ✓ Relationships already present, skipping.")

        # Step 4 — post-process (always run to ensure consolidation is fresh)
        self.stdout.write("  → Post-processing relationships...")
        success = consolidate_relationships(epub.id)
        if not success:
            raise Exception("Relationship consolidation failed")
        relationship_count = epub.relationships.count()
        self.stdout.write(self.style.SUCCESS(
            f"  ✓ Consolidated to {relationship_count} relationships"
        ))

        self.stdout.write(self.style.SUCCESS("\n✓ Continue processing complete!"))