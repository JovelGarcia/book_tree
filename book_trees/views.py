from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import get_object_or_404
from .models import EpubFile
from .serializers import EpubSerializer


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