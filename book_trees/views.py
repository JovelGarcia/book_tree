from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import EpubFile
from .serializers import EpubSerializer

@api_view(['GET'])
def epub_list_api(request):
    epubs = EpubFile.objects.all()
    serializer = EpubSerializer(epubs, many=True)
    return Response(serializer.data)
