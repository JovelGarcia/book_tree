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
