"""---------------------------------------------------------------------------------------------------------------------
    IMPORTS
---------------------------------------------------------------------------------------------------------------------"""
# Django tools imports:
from django.http import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render
from django.contrib.auth import authenticate
from django.views.decorators.csrf import ensure_csrf_cookie, csrf_protect, csrf_exempt
from django.conf import settings

# App item imports:
from .models import POI, POIImage, Action, ActionImage, LoraModel, DispatchedLoraCodes, FECity, FEArea, FEPoiMarker
from .serializers import POISerializer, POIImageSerializer, ActionSerializer, LoraModelSerializer
from .forms import ImageUploadForm

# Utilities imports:
import uuid
from PIL import Image
from ast import literal_eval

# Scripts imports
import sys
sys.path.append('../scripts')
from scripts.generator import generate
from scripts.action_picker import pick_action
from scripts.training_dispatcher import dispatch

"""---------------------------------------------------------------------------------------------------------------------
    VIEW | GENERATE IMAGE
---------------------------------------------------------------------------------------------------------------------"""
"""
VIEW QUERY PARAMETERS:
host/diffusers_api/get_poi_images?
poi_name=<value> 
&poi_image_name=<value> 
&action_name=<value> 
&action_shot_type=<value> 
&age=<value> 
&gender=<value> 
&other_details=<value> 
&selected_lora=<value> 
&dynamic_action_selection=[true|false] 
&action_prompt=<value>

&is_background_edited
&theme_description
&weather_description
&is_theme_set
&is_weather_set
&is_night_time

# esempio: 
http://127.0.0.1:8000/diffusers_api/generate_image?poi_name=boboli&poi_image_name=tunnel&action_name=wave_hand_1&action_shot_type=CLS&age=young&gender=woman&other_details=red%20shirt&selected_lora=gladdo
"""

def GenerateImage(request):
    
    try:
        poi_name = request.GET.get("poi_name")
        poi_image_name = request.GET.get("poi_image_name")
        poi_obj = POI.objects.get(name=poi_name)
        poi_image_obj = POIImage.objects.get(name=poi_image_name)

        # Seleziona l'azione che dev'essere rappresentata nell'immagine:
        action_name = ""
        if(request.GET.get("dynamic_action_selection")=="true"):
            action_name = pick_action(request.GET.get("action_prompt"))
        else:
            action_name = request.GET.get("action_name")
        action_obj = Action.objects.get(name=action_name)
       
        # Dell'azione scelta prendi la foto con lo shot_type selezionato:
        selected_shot_type = request.GET.get("action_shot_type")
        action_images = ActionImage.objects.filter(action=action_obj)
        action_image_obj = action_images.get(shot_type=selected_shot_type)

    except (POI.DoesNotExist, POIImage.DoesNotExist, Action.DoesNotExist, ActionImage.DoesNotExist) as error:
        log = " !! SOME SELECTED ITEM WAS NOT FOUND !! Error: " + error
        print(log)
        return HttpResponse(log)

    # Seleziona un'eventuale LoRA con cui generare le immagini
    using_lora = False
    selected_lora_id = request.GET.get("selected_lora")
    selected_lora = ""
    if(LoraModel.objects.filter(name=selected_lora_id).exists()):
        using_lora = True
        selected_lora = LoraModel.objects.get(name=selected_lora_id)
    
    # Alcuni elementi necessari per costruire il prompt
    age = request.GET.get("age")
    gender = request.GET.get("gender")
    other_details = request.GET.get("other_details")

    # Configurazioni di editing per il background
    is_background_edited_str = request.GET.get("is_background_edited")
    is_background_edited = False
    background_editing_config = {}

    if(is_background_edited_str == "true"):
        is_background_edited = True

        is_weather_set = False
        is_theme_set = False
        is_night_time = False

        if(request.GET.get("is_weather_set") == "true"):
            is_weather_set = True
        if(request.GET.get("is_theme_set") == "true"):
            is_theme_set = True
        if(request.GET.get("is_night_time") == "true"):
            is_night_time = True

        background_editing_config = {
            'theme_description' : request.GET.get("theme_description"),
            'weather_description' : request.GET.get("weather_description"),
            'is_weather_set' : is_weather_set,
            'is_theme_set' : is_theme_set,
            'is_night_time' : is_night_time
        }

    # Chiamata allo script di generazione
    """
    # PER GENERATORE CLASSICO
    image = generate(poi_obj, poi_image_obj, action_obj, action_image_obj, age, gender, other_details, using_lora, selected_lora)
    unique_file_name =  "" + str(uuid.uuid4())[:8] + ".jpg"
    image.save(settings.MEDIA_ROOT + "/outputs/" + unique_file_name)
    print("generated image: " + settings.MEDIA_ROOT + "/outputs/" + unique_file_name)
    """

    #For ram free:
    unique_file_name = generate(poi_obj, poi_image_obj, action_obj, action_image_obj, age, gender, other_details, using_lora, selected_lora, is_background_edited, background_editing_config)
    url_json = { "url" : settings.MEDIA_URL + "outputs/" + unique_file_name }

    return JsonResponse(url_json, safe=False)

"""---------------------------------------------------------------------------------------------------------------------
    VIEW | DB SUMMARY (GET)
---------------------------------------------------------------------------------------------------------------------"""
"""
Questa vista restituisce un JSON contenente i serializer delle entry delle seguenti tabelle:
- POI: Restituisce i nomi dei Point Of Interest disponibili
- POIImage: Restituisce per ogni immagine il POI a cui è associato, il nome dell'immagine e l'url statica a cui trovarla
- Action: Restituisce i nomi delle azioni disponibili 
Viene utilizzato dal front end per mostrare le scelte disponibili all'utente.
"""

def DBSummary(request):
    # Serializza tutti gli elementi nel POI Model
    poi_objects = POI.objects.all()
    poi_objects_serializer = POISerializer(poi_objects, many=True)

    # Serializza tutti gli elementi nel POIImage Model
    poi_image_objects = POIImage.objects.all()
    poi_image_objects_serializer = POIImageSerializer(poi_image_objects, many=True)

    # Serializza tutti gli elementi nell'Action Model
    action_objects = Action.objects.all()
    action_objects_serializer = ActionSerializer(action_objects, many=True)

    # Prepara il dizionario di ritorno
    serializer_data = {"pois" : poi_objects_serializer.data, "poi_images" : poi_image_objects_serializer.data, 
    "actions" : action_objects_serializer.data}
   
    return JsonResponse(serializer_data, safe=False)

def GetActions(request):
    action_objects = Action.objects.all()
    action_objects_serializer = ActionSerializer(action_objects, many=True)

    serializer_data = {"actions" : action_objects_serializer.data}

    return JsonResponse(serializer_data, safe=False) 


"""---------------------------------------------------------------------------------------------------------------------
    VIEW | POI IMAGES FROM POI NAME (GET)
---------------------------------------------------------------------------------------------------------------------"""
"""
Questa vista restituisce tutte le immagini relative ad un point of interest
"""
"url/get_poi_images?poi=poi_name"

def GetPoiImages(request):
    poi_object = POI.objects.get(name=request.GET.get("poi"))
    poi_image_objects = POIImage.objects.filter(poi=poi_object)
    poi_image_objects_serializer = POIImageSerializer(poi_image_objects, many=True)

    serializer_data = {"poi_images" : poi_image_objects_serializer.data}
   
    return JsonResponse(serializer_data, safe=False)

"""---------------------------------------------------------------------------------------------------------------------
    VIEW | LORA TRAINING DISPATCHER (GET|POST)
---------------------------------------------------------------------------------------------------------------------"""
"""
Se questa vista viene chiamata con il metodo GET allora restituisce un form per fare un POST di 5 immagini, quelle che vengono
utilizzate per il training.
Se invece è chiamata con il metodo POST e nel body sono presenti le 5 immagini, allora da il via al training del modello LoRA 
e restituisce all'utente un codice ad esso associato (che potrà utilizzare una volta che il modello è pronto).
"""

@csrf_exempt
def LoraTraining(request):

    if(request.method=='GET'):
        context = {}
        context['form'] = ImageUploadForm()
        return render( request, "images_upload_form.html", context)
        
    # Load posted images in PIL Image objects
    try:
        image = request.FILES['image1']
        image1 = Image.open(image).convert('RGB')
        image = request.FILES['image2']
        image2 = Image.open(image).convert('RGB')
        image = request.FILES['image3']
        image3 = Image.open(image).convert('RGB')
        image = request.FILES['image4']
        image4 = Image.open(image).convert('RGB')
        image = request.FILES['image5']
        image5 = Image.open(image).convert('RGB')
    except:
        print(" !! TRAINING IMAGE POST ERROR !! ")
        return HttpResponse("!! Error while uploading images, try again !!")
    
    images = [image1,image2,image3,image4,image5]

    # Generate unique name (id) for the new LoRA model
    lora_id = str(uuid.uuid4())[:8]
    query_set = LoraModel.objects.filter(name=lora_id)
    while(query_set.exists()):
        lora_id = str(uuid.uuid4())[:8]
        query_set = LoraModel.objects.filter(name=lora_id)
    dispatch(lora_id, images)

    #context = {'code' : lora_id}
    #return render( request, "images_submitted.html", context)

    DispatchedLoraCodes.objects.create(code=lora_id)

    response_data = {"lora_code" : lora_id}
   
    return JsonResponse(response_data, safe=False)

"""---------------------------------------------------------------------------------------------------------------------
    VIEW | LORA MODEL POST (POST)
---------------------------------------------------------------------------------------------------------------------"""
"""
Questa vista serve per fare il POST di un modello LoRA; viene chiamata automaticamente dallo script di training quando
ha finito di addestrare il LoRA per caricarlo sul database.
L'esecuzione del POST è consentita solo ad un admin che abbia fornito username e password nella richiesta di POST 
"""

@ensure_csrf_cookie
def PostLoraModel(request):
    if( request.method=='POST'):
        if( 'username' not in request.POST.keys()) or ( 'password' not in request.POST.keys()):
            return HttpResponse("POST data missing") 
        user = authenticate(username=request.POST['username'], password=request.POST['password'])
        if (user is not None and user.is_superuser):
            print("AUTENTICATED ADMIN - Proceding to upload lora model")
            lora_model_name = request.POST['lora_model_name']
            lora_model = request.FILES['lora_model']
            lora_item = LoraModel.objects.create(name=lora_model_name, model_path=lora_model, model_trigger="xkywkrav")
            lora_dispatched_code = DispatchedLoraCodes.objects.get(code=lora_model_name)
            lora_dispatched_code.delete()
            return HttpResponse("POST VALUE IS:")
        print("USER NOT ALLOWED TO POST")
        return HttpResponse("Not allowed to POST")
    return HttpResponse("POST only view")

"""---------------------------------------------------------------------------------------------------------------------
    VIEW | CHECK LORA CODE EXISTENCE (GET)
---------------------------------------------------------------------------------------------------------------------"""
"""
Questa vista consente di checkare lo stato del training del modello LoRA
"""

def CheckLoraCode(request):
    code = request.GET.get('code')

    if LoraModel.objects.filter(name=code).exists():
        response_data = {"result" : "Ready to use"}
        return JsonResponse(response_data, safe=False)

    if DispatchedLoraCodes.objects.filter(code=code).exists():
        response_data = {"result" : "Not ready yet"}
        return JsonResponse(response_data, safe=False)

    response_data = {"result" : "Code doesn't exsists"}
    return JsonResponse(response_data, safe=False)

"""---------------------------------------------------------------------------------------------------------------------
    VIEW | GET MAPS DATA (GET)
---------------------------------------------------------------------------------------------------------------------"""
"""
Questa vista consente di checkare lo stato del training del modello LoRA
"""

def GetMapsData(request):

    cities = FECity.objects.all()
    json_city_data = {}

    for city in cities:

        areas = FEArea.objects.filter(city=city)
        city_areas = []

        for area in areas:

            markers = FEPoiMarker.objects.filter(city=city, area_number=area.area_number)
            markers_data = []

            for marker in markers:

                markers_data.append({
                    "position": {"x": marker.x_pos, "y": marker.y_pos},
                    "value": marker.poi.name,
                    "popup_image": "http://" + request.get_host() + marker.overview_image.url,
                })

            city_areas.append({ 
                "canvas_position": {"x": area.canvas_x_pos, "y": area.canvas_y_pos},
                "markers" : markers_data,
                "background_image": "http://" + request.get_host() + area.area_image.url,
                "connected_areas": literal_eval(area.connected_areas),
            })

                
        json_city_data[city.city_name] = city_areas

    return JsonResponse(json_city_data, safe=False)





