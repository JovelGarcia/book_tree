import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'book_tree.settings')
django.setup()

from book_trees.processing import extract_chapters_from_epub, extract_characters_with_chunks, analyze_chunk_with_llm

# Trial 1 - Proper noun implicit - PASSED

# chunk = {
#     'context': 'A delicious fragrance wafted upward—roasted pork and crispy bacon and apples glazed in a rich wine sauce, resting on a bed of browned potatoes. Beside it was a bowl of fresh peas, swimming in butter seasoned with tarragon from the garden. And of course there was the baguette Vianne had made yesterday morning. As always, Sophie talked all through supper. She was like her Tante Isabelle in that way—a girl who couldn’t hold her tongue.',
#     'characters_in_context': ['Tante Isabelle', 'Vianne'],
#     'chapter_number': 2
# }
#
# relationships = analyze_chunk_with_llm(chunk, api_key="AIzaSyA67H6W1qQJhICZOloYeyAs-XnM-TeUIyY")
# print(relationships)
#
# [{'character_1': 'Sophie', 'character_2': 'Tante Isabelle', 'relationship_type': 'family',
#   'specific_type': 'aunt/niece', 'confidence': 0.9,
#   'evidence': 'She was like her Tante Isabelle in that way—a girl who couldn’t hold her tongue.', 'chapter_number': 2}]

# Trial 2 - False positive? - PASSED

# chunk = {
#     'context': 'Can’t you hear that? There’s someone knocking at the door.” Vianne shook her head (all she’d heard was the thunk-thunk-thunk of the axe) and went to the door, opening it. Rachel stood there, with the baby in her arms and Sarah tucked in close to her side. “You are teaching today with your hair pinned?” “Oh!”',
#     'characters_in_context': ['Rachel', 'Sarah', 'Vianne'],
#     'chapter_number': 10
# }
#
# relationships = analyze_chunk_with_llm(chunk, api_key="AIzaSyA67H6W1qQJhICZOloYeyAs-XnM-TeUIyY")
# print(relationships)
#
# [{'character_1': 'Rachel', 'character_2': 'Sarah',
#   'relationship_type': 'family', 'specific_type': 'mother/child',
#   'confidence': 0.8, 'evidence': 'Rachel stood there, with the baby in her arms and Sarah tucked in close to her side.',
#   'chapter_number': 10},
#  {'character_1': 'Rachel', 'character_2': 'Vianne',
#   'relationship_type': 'other', 'specific_type': 'acquaintance/neighbor',
#   'confidence': 0.6, 'evidence': 'Rachel stood there, with the baby in her arms and Sarah tucked in close to her side. “You are teaching today with your hair pinned?”',
#   'chapter_number': 10}]

import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'book_tree.settings')
django.setup()

from book_trees.processing import (
    extract_characters_with_chunks,
    find_character_name_variations,
    deduplicate_characters,
    analyze_chunk_with_llm
)
from book_trees.models import Character

API_KEY = 

def test_deduplication(epub_id):
    """Test how much deduplication reduces character count"""

    print("\n" + "=" * 60)
    print("CHARACTER DEDUPLICATION TEST")
    print("=" * 60)

    before = Character.objects.filter(epub_id=epub_id).count()
    print(f"\nBefore deduplication: {before} characters")

    # Show what will be merged
    groups = find_character_name_variations(epub_id)
    print(f"\nFound {len(groups)} groups to merge:")
    for group in groups[:5]:  # Show first 5
        names = [c.name for c in group]
        print(f"  → {' | '.join(names)}")
    if len(groups) > 5:
        print(f"  ... and {len(groups) - 5} more")

    # Actually merge
    stats = deduplicate_characters(epub_id)

    print(f"\nRESULT: Reduced from {before} → {stats['final_count']} characters")
    print(f"   Eliminated {stats['reduction']} duplicates ({stats['reduction'] / before * 100:.1f}%)")


def test_relationships():
    """Test relationship extraction on 3 quick examples"""

    print("\n" + "=" * 60)
    print("RELATIONSHIP EXTRACTION TEST")
    print("=" * 60)

    tests = [
        {
            'desc': 'Family relationship',
            'context': 'Sophie was like her Tante Isabelle—a girl who couldn\'t hold her tongue.',
            'chars': ['Sophie', 'Tante Isabelle'],
            'expect': 'family'
        },
        {
            'desc': 'Parent-child',
            'context': 'Rachel stood with the baby in her arms and Sarah tucked close to her side.',
            'chars': ['Rachel', 'Sarah'],
            'expect': 'family'
        },
        {
            'desc': 'No relationship',
            'context': 'The waiter brought the menu. John glanced at Mary briefly.',
            'chars': ['John', 'Mary'],
            'expect': 'none'
        }
    ]

    passed = 0
    for i, test in enumerate(tests, 1):
        print(f"\n{i}. {test['desc']}")
        print(f"   Expected: {test['expect']}")

        chunk = {
            'context': test['context'],
            'characters_in_context': test['chars'],
            'chapter_number': 1
        }

        rels = analyze_chunk_with_llm(chunk, API_KEY)

        if rels:
            rel = rels[0]
            print(f"   Found: {rel['relationship_type']} ({rel['specific_type']})")
            print(f"   Confidence: {rel['confidence']}")
            if rel['relationship_type'] == test['expect']:
                print(f"PASS")
                passed += 1
            else:
                print(f"FAIL")
        else:
            print(f"   Found: none")
            if test['expect'] == 'none':
                print(f"PASS")
                passed += 1
            else:
                print(f"FAIL")

    print(f"\nRESULT: {passed}/3 tests passed ({passed / 3 * 100:.0f}%)")


if __name__ == "__main__":
    print("\nQUICK FEATURE TEST")
    print("=" * 60)

    epub_id = int(input("Enter EPUB ID (default=1): ") or "1")

    print("\n1. Test deduplication only (free, fast)")
    print("2. Test relationship extraction only (uses API)")
    print("3. Test both")

    choice = input("\nChoice (default=1): ") or "1"

    if choice in ["1", "3"]:
        test_deduplication(epub_id)

    if choice in ["2", "3"]:
        test_relationships()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60 + "\n")
