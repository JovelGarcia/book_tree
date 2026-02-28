from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from .models import EpubFile, Character, Relationship
from .serializers import EpubSerializer, CharacterSerializer, RelationshipSerializer


@api_view(['GET'])
def epub_list_api(request):
    epubs = EpubFile.objects.all()
    serializer = EpubSerializer(epubs, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def epub_api(request, id):
    epub = get_object_or_404(EpubFile, id=id)
    serializer = EpubSerializer(epub, many=False)
    return Response(serializer.data)


@api_view(['POST'])
@parser_classes([MultiPartParser, FormParser])
def epub_upload_api(request):
    file = request.FILES.get('file')

    if not file:
        return Response({'error': 'No file provided.'}, status=status.HTTP_400_BAD_REQUEST)

    if not file.name.endswith('.epub'):
        return Response({'error': 'File must be an .epub.'}, status=status.HTTP_400_BAD_REQUEST)

    if file.size > 50 * 1024 * 1024:
        return Response({'error': 'File must be under 50MB.'}, status=status.HTTP_400_BAD_REQUEST)

    epub = EpubFile(file=file, original_filename=file.name)
    epub.save()

    serializer = EpubSerializer(epub)
    return Response(serializer.data, status=status.HTTP_201_CREATED)


# --- Characters ---

@api_view(['GET', 'POST'])
def character_list_api(request, epub_id):
    epub = get_object_or_404(EpubFile, id=epub_id)

    if request.method == 'GET':
        characters = Character.objects.filter(epub=epub)
        serializer = CharacterSerializer(characters, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = CharacterSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(epub=epub)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PUT', 'DELETE'])
def character_detail_api(request, epub_id, character_id):
    epub = get_object_or_404(EpubFile, id=epub_id)
    character = get_object_or_404(Character, id=character_id, epub=epub)

    if request.method == 'GET':
        serializer = CharacterSerializer(character)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = CharacterSerializer(character, data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        character.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)


# --- Relationships ---

@api_view(['GET', 'POST'])
def relationship_list_api(request, epub_id):
    epub = get_object_or_404(EpubFile, id=epub_id)

    if request.method == 'GET':
        relationships = Relationship.objects.filter(epub=epub).select_related(
            'character_1', 'character_2'
        )
        serializer = RelationshipSerializer(relationships, many=True)
        return Response(serializer.data)

    elif request.method == 'POST':
        serializer = RelationshipSerializer(data=request.data, context={'epub': epub})
        if serializer.is_valid():
            serializer.save(epub=epub)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['GET', 'PUT', 'DELETE'])
def relationship_detail_api(request, epub_id, relationship_id):
    epub = get_object_or_404(EpubFile, id=epub_id)
    relationship = get_object_or_404(Relationship, id=relationship_id, epub=epub)

    if request.method == 'GET':
        serializer = RelationshipSerializer(relationship)
        return Response(serializer.data)

    elif request.method == 'PUT':
        serializer = RelationshipSerializer(relationship, data=request.data, context={'epub': epub})
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    elif request.method == 'DELETE':
        relationship.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)