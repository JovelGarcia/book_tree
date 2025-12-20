"""
Management command to process pending EPUB files

Usage:
    python3 manage.py process_epubs
    python manage.py process_epubs --epub-id 5
"""

from django.core.management.base import BaseCommand
from book_trees.models import EpubFile
from book_trees.processing import process_epub_file

class Command(BaseCommand):
    help = 'Process pending EPUB files to extract chapters'

    def add_arguments(self, parser):
        parser.add_argument(
            '--epub-id',
            type=int,
            help='Process a specific EPUB by ID',
        )
        parser.add_argument(
            '--reprocess',
            action='store_true',
            help='Reprocess EPUBs even if already completed'
        )

    def handle(self, *args, **options):
        epub_id = options.get('epub_id')
        reprocess = options.get('reprocess')

        if epub_id:
            try:
                epub = EpubFile.objects.get(id=epub_id)
                self.stdout.write(f"Processing EPUB: {epub.original_filename}")

                if reprocess:
                    epub.characters.all().delete()
                    epub.chapters.all().delete()

                success = process_epub_file(epub.id)

                if success:
                    self.stdout.write(self.style.SUCCESS(f"Successfully processed {epub.original_filename}"))
                else:
                    self.stdout.write(self.style.ERROR(f"Failed to process {epub.original_filename}"))

            except EpubFile.DoesNotExist:
                self.stdout.ERROR(f"EPUB with ID {epub_id} not found")

        else:
            if reprocess:
                epubs = EpubFile.objects.all()
                self.stdout.write(f"Processing all {epubs.count} EPUBs...")
            else:
                epubs = EpubFile.objects.filter(status='p')
                self.stdout.write(f"Processing {epubs.count()} pending EPUBs...")

            success_count = 0
            fail_count = 0

            for epub in epubs:
                self.stdout.write(f"Processing {epub.original_filename}")

                if reprocess:
                    epub.characters.all().delete()
                    epub.chapters.all().delete()
                    epub.status = 'p'
                    epub.save()

                try:
                    success = process_epub_file(epub.id)

                    if success:
                        success_count += 1
                        self.stdout.write(f"Success")

                    else:
                        fail_count += 1
                        # self.stdout.write(f"Failed: {epub.error_message}")

                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"  ✗ Failed: {str(e)}"))
                    fail_count += 1

            self.stdout.write(self.style.SUCCESS(f"\n\nCompleted: {success_count} successful, {fail_count} failed"))