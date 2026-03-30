from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from .models import MediaRequest, Character, Relationship
from .serializers import MediaRequestSerializer, CharacterSerializer, RelationshipSerializer
from .agent import run_media_agent

@api_view(['GET'])
def media_list_api(request):
    qs = MediaRequest.objects.all()
    return Response(MediaRequestSerializer(qs, many=True).data)


@api_view(['POST'])
def media_submit_api(request):
    serializer = MediaRequestSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    instance = serializer.save()
    run_media_agent(instance.id)   # synchronous for now

    return Response(MediaRequestSerializer(instance).data, status=status.HTTP_201_CREATED)

@api_view(['GET'])
def media_detail_api(request, id):
    try:
        media = MediaRequest.objects.get(id=id)
    except MediaRequest.DoesNotExist:
        return Response({'error': 'Not found.'}, status=status.HTTP_404_NOT_FOUND)
    return Response(MediaRequestSerializer(media).data)


@api_view(['GET'])
def media_graph_api(request, id):
    """Returns D3-ready { nodes, edges } once processing is complete."""
    try:
        media = MediaRequest.objects.get(id=id)
    except MediaRequest.DoesNotExist:
        return Response({'error': 'Not found.'}, status=status.HTTP_404_NOT_FOUND)

    if media.status != 'c':
        return Response(
            {'error': 'Graph not ready.', 'status': media.status},
            status=status.HTTP_202_ACCEPTED,
        )

    nodes = CharacterSerializer(media.characters.all(), many=True).data
    edges = RelationshipSerializer(media.relationships.all(), many=True).data
    return Response({'nodes': nodes, 'edges': edges})


@api_view(['GET'])
def character_list_api(request, id):
    try:
        media = MediaRequest.objects.get(id=id)
    except MediaRequest.DoesNotExist:
        return Response({'error': 'Not found.'}, status=status.HTTP_404_NOT_FOUND)
    return Response(CharacterSerializer(media.characters.all(), many=True).data)


@api_view(['GET'])
def relationship_list_api(request, id):
    try:
        media = MediaRequest.objects.get(id=id)
    except MediaRequest.DoesNotExist:
        return Response({'error': 'Not found.'}, status=status.HTTP_404_NOT_FOUND)
    return Response(RelationshipSerializer(media.relationships.all(), many=True).data)