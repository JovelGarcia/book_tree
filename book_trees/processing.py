import ebooklib
import re
import spacy
import json
from ebooklib import epub
from bs4 import BeautifulSoup
from django.db import transaction
from .models import EpubFile, Chapter, Character, Relationship

# load NLP
nlp = spacy.load("en_core_web_sm")

def extract_chapters_from_epub(epub_path):
    """Extract chapters and text from an EPUB file."""
    book = epub.read_epub(epub_path)
    chapters = []

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text("\n", strip=True)

            if "chapter" in item.get_name().lower():
                match = re.search(r'chapter[_\s-]*(\d+)', item.get_name().lower())
                chapter_number = int(match.group(1)) if match else None

                chapters.append({
                    'chapter_number': chapter_number,
                    'title': item.get_name(),
                    'content': text
                })

    return chapters

def process_epub_file(epub_id):

    epub = None
    try:
        epub = EpubFile.objects.get(id=epub_id)
        epub.status = 'pr'
        epub.save()

        file_path = epub.file.path

        chapters_data = extract_chapters_from_epub(file_path)

        for chapter_data in chapters_data:
            Chapter.objects.create(
                epub = epub,
                title=chapter_data['title'],
                content = chapter_data['content'],
                chapter_number = chapter_data['chapter_number']

            )

        epub.status = 'c'
        epub.processed = True
        epub.save()

        return True



    except Exception as e:
        if epub:
            epub.status = 'f'
            epub.error_message = str(e)
            epub.save()
        raise


def extract_characters_with_chunks(epub_id, context_sentences=2):
    """
    Extracts character names from EPUB using spaCy NER and creates chunks with context
    Avoids overlapping chunks by advancing past the context window

    Args:
        epub_id: ID of the EpubFile to process
        context_sentences: Number of sentences before/after to include in chunk

    Returns:
        Number of unique characters found
    """
    epub = EpubFile.objects.get(id=epub_id)
    chapters = epub.chapters.all()

    character_counts = {}
    first_appearance = {}

    for chapter in chapters:
        doc = nlp(chapter.content)

        sentences = list(doc.sents)
        annotated_chunks = []

        i = 0
        while i < len(sentences):
            sent = sentences[i]
            entities_in_sentence = []

            for ent in sent.ents:
                if ent.label_ == 'PERSON':
                    name = ent.text.strip()
                    name = name.rstrip("'s")

                    # skip single letters, common false positive
                    if len(name) <= 1:
                        continue

                    entities_in_sentence.append(name)

                    character_counts[name] = character_counts.get(name, 0) + 1

                    if name not in first_appearance:
                        first_appearance[name] = chapter.chapter_number

            if entities_in_sentence:
                start_idx = max(0, i - context_sentences)
                end_idx = min(len(sentences), i + context_sentences + 1)

                context_sents = sentences[start_idx:end_idx]
                context_text = ' '.join([s.text.strip() for s in context_sents])

                all_characters_in_context = set(entities_in_sentence)
                for context_sent in context_sents:
                    for ent in context_sent.ents:
                        if ent.label_ == 'PERSON':
                            char_name = ent.text.strip().rstrip("'s")
                            if len(char_name) > 1:
                                all_characters_in_context.add(char_name)

                annotated_chunks.append({
                    'center_sentence': sent.text.strip(),
                    'context': context_text,
                    'characters_in_sentence': entities_in_sentence,
                    'characters_in_context': list(all_characters_in_context),
                    'sentence_index': i
                })

                i = end_idx
            else:
                i += 1

        chapter.annotated_sentences = annotated_chunks
        chapter.save()

    for name, count in character_counts.items():
        Character.objects.update_or_create(
            epub=epub,
            name=name,
            defaults={
                'mention_count': count,
                'first_appearance_chapter': first_appearance[name]
            }
        )

    return len(character_counts)


def extract_characters_simple(epub_id):
    """
    LEGACY FUNCTION

    Extracts character names from EPUBn using spaCy NER

    Args:
        epub_id: ID of the EpubFile to process

    Returns:
        Number of unique characters found
    """
    epub = EpubFile.objects.get(id=epub_id)
    chapters = epub.chapters.all()

    character_counts = {}
    first_appearance = {}


    for chapter in chapters:
        doc = nlp(chapter.content)
        annotated_sentences = []

        for sent in doc.sents:
            entities_in_sentence = []

            for ent in sent.ents:
                if ent.label_ == 'PERSON':
                    name = ent.text.strip()
                    name = name.rstrip("'s")

                    #skip single letters, common false positive
                    if len(name) <= 1:
                        continue

                    entities_in_sentence.append(name)

                    character_counts[name] = character_counts.get(name, 0) + 1

                    if name not in first_appearance:
                        first_appearance[name] = chapter.chapter_number

            if entities_in_sentence:
                annotated_sentences.append({
                    'text': sent.text.strip(),
                    'characters': entities_in_sentence
                })

        chapter.annotated_sentences = annotated_sentences
        chapter.save()

    for name, count in character_counts.items():
        Character.objects.update_or_create(
            epub = epub,
            name = name,
            defaults = {
                'mention_count': count,
                'first_appearance_chapter': first_appearance[name]
            }
        )

    return len(character_counts)



def extract_relationships_simple(epub_id):
    """
    LEGACY FUNCTION
    """
    epub = EpubFile.objects.get(id=epub_id)
    chapters = epub.chapters.all()

    relationship_keywords = {
        'brother': ['brother', 'bro'],
        'sister': ['sister', 'sis'],
        'cousin': ['cousin'],
        'father': ['father', 'dad', 'papa', 'daddy'],
        'mother': ['mother', 'mom', 'mama', 'mommy'],
        'uncle': ['uncle', 'unc'],
        'aunt': ['aunt', 'auntie']
    }

    relationships_found = 0

    for chapter in chapters:
        for sentence_data in chapter.annotated_sentences:
            characters = sentence_data['characters']
            text = sentence_data['text']

            if len(characters)  >= 2:
                for relationship_type, keywords in relationship_keywords.items():
                    for keyword in keywords:

                        pattern = rf"\b{keyword}(?:'s)?\b"
                        if re.search(pattern, text, re.IGNORECASE):
                            other_chars = [c for c in characters if c not in [characters[0], characters[1]]]

                            try:
                                char1_object = Character.objects.get(epub=epub, name=characters[0])
                                char2_object = Character.objects.get(epub=epub, name=characters[1])

                            except Character.DoesNotExist:
                                continue

                            rel, created = Relationship.objects.get_or_create(
                                epub=epub,
                                character_1=char1_object,
                                character_2=char2_object,
                                relationship_type=relationship_type,
                                defaults={
                                    'confidence': 0.7,
                                    'evidence': []
                                }
                            )

                            new_evidence = {
                                'chapter': chapter.chapter_number,
                                'text': text,
                                'other_characters': other_chars
                            }

                            if new_evidence not in rel.evidence:
                                rel.evidence.append(new_evidence)

                                rel.confidence=min(0.95, 0.7 + len(rel.evidence) * 0.05)
                                rel.save()

                            relationships_found += 1
                            break

    return relationships_found








