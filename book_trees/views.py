from rest_framework.decorators import api_view
from rest_framework.response import Response
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
