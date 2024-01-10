# INSTALLAZIONE

### Tramite Compose

Con docker installato e la working directory nella folder del progetto, eseguire il comando:

&emsp;&emsp;&emsp;&emsp;docker compose -f compose.yaml up

### Tramite Dockerfile manualmente

Con docker installato e la working directory nella folder del progetto, eseguire i comandi:

- Per creare l'immagine specificata nel Dockerfile:
  
&emsp;&emsp;&emsp;&emsp;docker build -t image_name .

- Per avviare un container sull'immagine precedentemente creata e accederne alla shell: (l'opzione --gpus all serve per consentire al container di accedere alla GPU del sistema che esegue il container )

&emsp;&emsp;&emsp;&emsp;docker run -it --gpus all -p 8000:8000 image_name

- Attraverso la shell avviata precedentemente, per avviare il server dell'applicazione entrare con la working directory in app/djangoproject ed avviare il server col comando:
  
&emsp;&emsp;&emsp;&emsp;python3 manage.py runserver 0.0.0.0:8000


# SISTEMA

I tempi e le considerazioni fatte sono in relazione alle seguenti risorse:

- GPU: NVIDIA GeForce GTX 1060 (6GB GPU dedicated memory)
- RAM: 16 GB
- OS: Docker Desktop on Windows 10 with WSL2
- Drive: HDD (no SSD was available)
- CPU Intel Core i5-8600k @ 3.60GHz

L'esecuzione su container Docker richiede che al container venga fornito almeno 9GB di RAM (l'utilizzo medio della RAM è inferiore ma ci sono dei peak sulla richiesta di risorse durante alcune fasi della generazione, come specificato successivamente) 

### Aumentare la RAM fornita ad un container docker ( su windows con WSL2 ):

Nella folder C:\users\user_name creare un file .wslconfig con al suo interno le seguenti righe:

```
[wsl2]
memory=9GB
```

NB: Fare attenzione a non cedere troppa RAM: oltre un certo limite si vincola il SO a rallentare e questo rallenta di conseguenza anche l'esecuzione del container. 

## TRAINING LORA

### Flow di esecuzione

L'addestramento è eseguito su 5 immagini dell'utente su ciascuna delle quali viene fatto 300 iterazioni di training (per un totale di 1500 iterazioni); queste configurazioni di addestramento sono ritenute minime per quanto riguarda la cultura del fine tuning con LoRA.

L'addestramento è ottenuto facendo uso degli script nel seguente repository [1] e nell'applicazione segue il seguente flow:

- L'utente seleziona nel front-end le 5 immagini con cui creare il modello LoRA e le comunica alla vista LoraTraining nell'app diffuser_api del server Django
- La vista LoraTraining controlla che siano presenti le 5 immagini nel body
- Se il controllo va a buon fine la vista genera un codice di 8 cifre (LoraCode); tale codice è comunicato:
  - allo script training_dispatcher che lancia un subprocess dove viene effettuato il training
  - all'utente che poi lo utilizzerà per fare riferimento al proprio modello LoRA una volta che questo è stato ultimato

Il dispatcher che lancia l'addestramento esegue i seguenti passi:

- crea una folder associata all'utente per cui si sta eseguendo il training; a questa è assegnato come nome il codice LoRA appena generato e al suo interno viene predisposto le immagini dell'utente per il successivo training  
- avvia un subprocess per l'esecuzione dello script launch.bash presente nella folder training-app

Lo script launch.bash:

- contiene il comando per l'effettivo avvio del training implementato dallo script train_network.py nella folder kohya-scripts; eventuali configurazioni del training sono da fare nei parametri di questo comando.
- Quando il training è ultimato, launch.bash esegue lo script post-back.py che manda un comando di POST al server Django richiedendo di aggiungere il modello LoRA appena addestrato ai modelli LoRA disponibili sull'applicazione.
- Infine provvede a rimuovere i dati utilizzati per il training dell'utente appena servito (sia i dati di input che quelli di output dato che adesso sono caricati sul server django).

NB: SE SI MODIFICA QUESTO SCRIPT SU WINDOWS, ASSICURARSI CHE SUCCESSIVAMENTE SIA CONVERTITO NELLA FORMATTAZIONE UNIX PER EVITARE INUTILI MAL DI TESTA ( altrimenti durante l'esecuzione del training viene fuori errori a cui è difficile risalire, quando il problema era semplicemente la formattazione dello script ) 

### Profiling

I tempi medi di esecuzione del training nel sistema di riferimento vanno dai 60 ai 90 minuti; l'uso delle risorse è stabile (non cè nessun particolare picco sulla richiesta di risorse)

## GENERAZIONE

### Flow di esecuzione

Per gli istrumenti di inferenza (generazione) è stato fatto uso della libreria diffusers [2].

#### Interfaccia dell'API

Si accede alla generazione di un'immagine attraverso la vista GenerateImage dell'API a cui devono essere forniti i seguenti parametri:

- poi_name: nome del punto d'interesse contenete l'immagine su cui iniziare la generazione del soggetto
- poi_image: nome dell'immagine di background da cui iniziare la generazione; deve combaciare con il nome di una immagine nel database
- action_name: nome dell'azione da utilizzare per il soggetto generato; deve combaciare con il nome di un'azione nel database
- action_shot_type: a che distanza è inserito il soggetto nell'immagine; è necessario fornire uno dei seguenti valori: "CLS", "MES", "FUS" che Rispettivamente indicano: close, medium distance, far.
- age: deve indicare l'età del soggetto attraverso uno dei seguenti aggettivi: "young", "adult", "old"
- gender: deve indicare il sesso del soggetto attraverso uno dei seguenti valori: "man", "woman"
- other_details: deve indicare dettagli sul soggetto; viene inserito nel prompt di generazione con la struttura "wearing" + other_details
- selected_lora: può contenere o meno un LoRA code; se non contiene nessun codice o un codice errato, semplicemente la generazione avviene senza l'inserimento del soggetto, altrimenti viene eseguito pure quest'ultimo step.
- dynamic_action_selection: deve indicare true o false; nel primo caso l'action_name scelto viene sovrascritto dalla selezione dinamica della posa fatta dallo script action_picker.py, altrimenti è utilizzata l'action_name scelta dall'utente
- action_prompt: contiene la descrizione dell'azione che viene eventualmente utilizzata dallo script action_picker.py per la selezione dinamica.

#### Esecuzione della vista di generazione (GenerateImage in views.py)

Impostati i precedenti parametri, la vista procede nel seguente modo: 

- Viene selezionata l'immagine di background estrendo dal db l'oggetto poi_image_obj: questo contiene l'immagine da utilizzare e la descrizione di tale immagine (che viene utilizzata nella costruzione del prompt)
- Viene selezionata l'azione da utilizzare, tramite la scelta dinamica o manuale dell'utente, estraendo dal db l'oggetto action_obj: questo specifica il nome e la descrizione dell'azione selezionata
- A questo punto si prende tutte le immagini relative all'azione selezionata filtrando la table delle immagini con l'action_obj appena scelto; si ottiene una lista di immagini relative ad una stessa azione ma con differenti distanze del soggetto
- Si utilizza l'action_shot_type per scegliere definitivamente quali immagini dell'azione utilizzare e si mettono nell'oggetto action_image_obj: questo contiene
  - L'immagine del manichino utilizzato per guidare la generazione della posa
  - L'immagine per la maschera di inserimento della posa
  - L'immagine per la maschera di inserimento del volto
- Si controlla se l'eventuale codice LoRA fornito combacia con un modello esistente; in tal caso lo si utilizza per l'inserimento del volto dell'utente
- Quindi si lancia lo script di generazione; questo provvederà a mettere l'immagine di output nella folder data/outputs dandogli un nome generato casualmente (con un codice di 8 cire); tale codice è restituito alla vista che poi termina comunicandolo all'utente; l'utente potrà accedere in modo statico attraverso una request al server tramite un'url con la seguente struttura:
  
&emsp;&emsp;&emsp;&emsp; http://server_address:port/data/outputs/codice

### Profiling dei comandi nello script di generazione (generator.py)

Prima di passare all'implementazione dello script vediamo gli step computazionali più importanti per la generazione.

 Ad alto livello, per la generazione di una singola immagine, è necessario eseguire i seguenti step:

 - Download del modello da utilizzare
 - Setup della pipeline di generazione
 - Steps di inferenza della pipeline di generazione
 - Conversione dell'immagine da latent-space a immagine vera e propria (Decoder VAE)

In termini di comandi e risorse, tutto ciò viene implementanto nei seguenti passi:

- Chiamate alla funzione **from_petrained**: (Download del modello e setup della pipeline)

  Le chiamate a questa funzione, del tipo:

  &emsp;&emsp;&emsp;&emsp;pipe = StableDiffusion*Pipeline.from_pretrained("model_repository",...)

  Al primo avvio eseguono il download del modello specificato prendendolo dai repository di HuggingFace [3] e lo memorizzano nella cache di sistema; quindi provvedono a caricarlo in RAM. Nelle successive chiamate per uno stesso modello, questo è caricato in RAM raccogliendolo direttamente dalla cache di sistema; in corrispondenza di tali chiamate cè un elevato utilizzo del disco che può potenzialmente fare da bottleneck per il processo di generazione (specie nel caso di un HDD).

  La specifica dell'opzione torch_dtype=torch.float16 negli argomenti di questa funzione specifica il tipo di tipo di variabile in cui memorizzare i pesi del modello. Di default è utilizzato un float32; nel caso venga specificato si può invece dimezzare la richiesta di spazio utilizzando dei float16; questo impatta enormemente la velocità di esecuzione, specie nelle GPU low end (solo con questa opzione è possibile ridurre il tempo di inferenza di un fattore 10)

- Chiamate alla funzione **pipe.to("cuda")**: (setup della pipeline)
  
  Queste caricano il modello sulla memoria GPU e la configurano per essere eseguite col modello computazionale CUDA di NVIDIA (https://docs.nvidia.com/deploy/cuda-compatibility/).

- Chiamate alla funzione **pipe.enable_xformers_memory_efficient_attention()**: (setup della pipeline)

  Queste chiamate specificano di utilizzare determinati meccanismi di ottimizzazione nell'esecuzione dell'inferenza (per poter usare questa funzione è necessario aver installato il package python xformers nell'ambiente)

- Chiamate alla funzione **pipe(...)**: (Inferenza e conversione latent-space -> immagine)

  Queste chiamate, del tipo:

  &emsp;&emsp;&emsp;&emsp; result_img = pipe(...).image[0]

  Danno il via al vero e proprio algoritmo di inferenza; la loro esecuzione è localizzata sulla memoria della GPU che infatti, in corrispondenza di questi comandi, raggiunge livelli elevati (nel caso di 6GB cè un'utilizzo del 100%).

  Al termine dell'inferenza cè un picco nell'utilizzo della RAM, presupponibilmente per la conversione dell'immagine da latent space ad immagine vera e propria (non sono riuscito a trovare eventuale documentazione della libreria che giustifichi tale sforzo ).

### Esecuzione dello script

Esistono due script per la generazione, generator.py e generator_classic.py; il primo implementa gli stessi identici passi del secondo apportando delle ottimizzazioni per il motivo spiegato a breve, mentre il secondo è una descrizione sequenziale, più elengate, degli stessi passi computazionali; per questo motivo, per comprendere meglio ciò che viene eseguito, si fa riferimento a generator_classic.py. 

L'esecuzione dell'intero script si può riassumere nel seguente flow:

1. Si configura i prompt, specificandone una struttura sufficientemente funzionale da essere configurabile e allo stesso tempo triggerare le cose giuste durante l'inferenza; si riempie i campi di tale struttura utilizzando gli elementi passati come argomento alla funzione di generazione (quali descrizione del soggetto (age, gender, posa) e descrizione dell'ambiente (in poi_image))
2. Si effettua uno step di inpainting, utilizzando una pipeline ControlNet, per generare il soggetto sopra l'immagine di background scelta dal database; la pipeline ControlNet consente di combinare due meccanismi che guidano la generazione, quali:
  - Condizionamento della generazione a riprodurre la posa specificata utilizzando il modello openpose "lllyasviel/control_v11p_sd15_openpose"
  - Condizionamento della generazione per effettuare l'inpaiting del soggetto utilizzando il modello "runwayml/stable-diffusion-inpainting"
Grazie alla ControlNet è possibile combinare questi due passaggi in un'unico step di inferenza che combina inpainting e scelta della posa.
3. Si effettua uno step di inpaiting sul risultato della precedente operazione per aumentare i dettagli dell soggetto inpaintato; questo step lavora SOLO su un'area specifica dell'immagine, quella definita dalla maschera associata alla posa (ed in cui sarà sicuramente contenuto il soggetto generato precedentemente); si utilizza il modello "dreamlike-art/dreamlike-photoreal-2.0"
4. Si effettua dinuovo il punto 3
5. Si effettua adesso uno step di Img2Img che prende il risultato precedente ed effettua un'omogenizzazione della qualità dell'immagine
6. Se è stato scelto un LoRA, si fa un'ultimo step di inpainting per l'inserimento del volto del soggetto; anche in questo step di inpainting si va ad agire su una specifica maschera (in cui sarà sicuramente contenuto il volto del soggetto data la struttura di generazione); in questa fase è tuttavia presente un'ulteriore passaggio: per utilizzare il modello lora bisogna combinarlo con un modello pre-esistente; questo si ottiene con i seguenti steps:
    1. Si utilizza uno script della libreria diffusers (convert_lora_safetensor_to_diffusers) per combinare il modello LoRA dell'utente con il modello base "runwayml/stable-diffusion-inpainting".
    2. 2 Lo script per la combinazione carica in RAM e in GPU il modello "runwayml/stable-diffusion-inpainting" e lo combina con il modello LoRA inserendolo in un'oggetto StableDiffusionPipeline piuttosto che StableDiffusionInpaitingPipeline; dato che a noi serve il secondo tipo di pipeline (dato che il primo non consente l'inpaiting) diventa necessario salvare sul disco il modello e successivamente ricaricarlo col tipo appropiato di pipeline (purtroppo non ho trovato alternative a questo meccanismo utilizzando la libreria diffusers).

(Per la ricerca dietro alla scelta di questa struttura di generazione, vedere la tesi)

Ad ogni step viene salvato le immagini generate dentro tmp_data in una folder che ha come nome il codice prodotto all'inizio della generazione; al suo interno viene anche salvato temporaneamente il modello prodotto dallo step 6. Tale folder è eliminata a termine della generazione e l'immagine di output è memorizzata nella folder data/outputs con nome uguale al precedente codice.

### Profiling dei vari step

Come accennato precedentemente, i punti critici sono nei momenti in cui:

&emsp;&emsp;A. Si carica il modello per la lavorazione dello specifico step e se ne fa il setup

&emsp;&emsp;B. Si fa inferenza

&emsp;&emsp;C. Si genera l'immagine di lavorazione intermedia, si smonta la pipeline corrente e si passa allo step successivo

I punti A, B e C vengono ripetuti per ciascuno degli step 2, 3, 4, 5 e 6 ma con modalità differenti:

Gli step 2, 3, 4, 5 hanno grossomodo la stessa performance; per ciascuno, nel sistema di riferimento, si ha:

- A: Una volta che i rispettivi modelli sono in Cache, il setup della pipeline di generazione impiega circa 10 secondi e raggiunge valori di RAM che oscillano tra i 3GB ai 5 GB
- B: Una volta che l'inferenza è iniziata, il la generazione impiega generalmente dai 40 ai 60 secondi; a questo punto l'elaborazione è sulla GPU che raggiunge l'uso di tutti e 6 i GB di RAM
- C: A termine della generazione è impiegato dai 10 ai 20 secondi prima dell'inizio del punto A del successivo step; in questa fase l'utilizzo della RAM si aggira sempre attorno ai 3GB-5GB

Lo step 6 è invece quello più critico per i seguenti motivi:
1. A differenza degli altri step il modello da utilizzare va creato (ottenuto combinando il modello "runwayml/stable-diffusion-inpainting" con il LoRA dell'utente), memorizzato sul disco e ricaricato (come specificato in precedenza). 
2. Si fa uso di una risoluzione più alta dell'immagine generata: infatti con una risoluzione bassa l'algoritmo di generazione distorge i tratti del volto generato; questa problematica è arginata utilizzando un LoRA addestrato su immagini che contengono diverse risoluzioni del volto dell'utente.

  Tuttavia è l'utente a selezionare le immagini e non si può sempre fare affidamento a questa condizione; se il modello è addestrato su immagini in cui il volto è esteso su uno stesso numero di pixel, questo è incentivato a legare il concetto del volto a tale specifica dimensione. 
  
  Se ad esempio le immagini di training sono tutte addestrate su dei primi piani 1980x1080, quando poi il modello è utilizzato per generare il volto di una persona inquadrata che appare per intero (da testa a piedi) in una immagine 512x512, dove quindi il volto ricoprirà un'area di circa 50x50 pixel, il modello è incapace di sintetizzare il volto su tale dimensione ristretta. 

(l'idea intuitiva dietro a questo problema è che il modello, addestrato con tutti primi piani, lega la dimensione in pixel al concetto del volto; se invece si utilizza immagini con differenti dimensioni del volto allora il modello ne apprende le proporzioni: il concetto del volto diventa legato alle proporzioni piuttosto che alle specifiche dimensioni in pixel; di conseguenza la generazione è capace di sintetizzarlo a diverse risoluzioni)

In generale, durante i test e il profiling della generazione, è nello step 6 che si presentava maggior rischio che il container docker si chiudesse con errore 247 ( legato ad una richiesta di memoria maggiore a quella fornita) e che ha richiesto una aggiunta di RAM dedicata.

Tornando al profiling del punto 6 dunque si osserva:

- Per il merging del modello "runwayml/stable-diffusion-inpainting" con il LoRA:
  - Circa 10 secondi per caricare il modello "runwayml/stable-diffusion-inpainting" nella pipeline
  - Circa 30 secondi per fare il merge
  - Circa 60 secondi per salvare sul disco il modello creato
  In questa fase la RAM utilizzata raggiunge picchi di 7-8 GB e anche l'utilizzo del disco è molto elevato ( per via del salvataggio ) 
- Per la generazione
  - Circa 100 secondi per l'avvio dell'inferenza
  - Circa 4 minuti per completare l'inferenza con la risoluzione dell'immagine di output 1200x1200; qui l'utilizzo della ram scende e si mantiene attorno ai 5 GB.

    Il tempo per questo passaggio è drasticamente ridotto se si riduce la risoluzione di output (al costo della qualità dell'immagine finale)
    
  - Circa 2 minuti a termine dell'inferenza con picchi di ram fino a 8 GB
 
### Risultato riassuntivo

In generale la generazione di un'immagine, nel sistema di riferimento, richiede dai 10 ai 15 minuti; tuttavia i bottleneck principali sono l'avvio della pipeline piuttosto che l'inferenza stessa: sommando i tempi di inferenza dei vari step di media è speso 6 minuti complessivi, quantità che può essere drasticalmente ridotta riducendo la risoluzione dell'ultimo step, quello che poi ha di fatto i tempi di inferenza più lunghi.

In relazione al tempo di setup della pipeline è invece necessario fare la seguente osservazione: nel caso del sistema di riferimento si ha a disposizone una RAM contenuta e diventa dunque necessario alternare la presenza dei vari modelli in esecuzione; in una situazione di deploy, su macchine adeguate, ciascuno dei vari step può essere implementato su diversi hardware e ciascuna pipeline può rimanere costantemente pronta all'inferenza, riducendo teoricamente tempi in modo netto.

### Osservazione: generator.py vs generator_classic.py

Nonostante implementano gli stessi step computazionali, la necessità di generator.py rispetto a generator_classic.py nasce dal seguente problema riscontrato: quando si alterna i modelli di generazione in RAM la libreria diffusers produce dei Memory Leak; questi, gradualmente su diverse generazioni, rischiano di farla straripare.

Per ovviare a tale problema, che è possibile ancora riscontrare utilizzando lo script generator_classic.py, ho optato per la seguente strategia: gli step di generazione, dove carico i modelli in RAM, sono tutti eseguiti utilizzando dei subprocess; i vari step si comunicano le immagini semplicemente salvandole sul disco.

In questo modo, a prescindere dalle problematiche (inarginabili) di memory leaks inerenti alla libreria diffusers, terminato uno step di generazione viene chiuso il subproces all'interno del quale era stato caricato il modello in RAM; di conseguenza tutta la sua RAM è liberata assieme ad eventuali problemi di leak.

## TO-DO:

Un eventuale sviluppo può essere continuato dai seguenti task:

- Alla rimozione delle entries delle table delle azioni vengono eliminati anche le risorse associate a tale entry; questo non è ancora vero per le entries e le risorse delle PoiImages
- Cleanup delle risorse quando training o generazione terminano bruscamente: ad ora le risorse solo rimosse solo a fine dei relativi script e un'eventuale interruzione le dimentica sul sistema (ad esempio file in tmp_data o i file in input e output in traning-app)
- Eventualmente, su hardware adeguato, far si che lo script di generazione non debba chiudere e ricaricare le pipeline ogni volta.

## Resources Link

[1] Kohya: https://github.com/kohya-ss/sd-scripts

[2] Diffusers: https://huggingface.co/blog/stable_diffusion

[3] HuggingFace: https://huggingface.co/

Link ai modelli stable-diffusions utilizzati:

Openpose controlnet: https://huggingface.co/lllyasviel/control_v11p_sd15_openpose

Inpainting: https://huggingface.co/runwayml/stable-diffusion-inpainting

Detailing: https://huggingface.co/dreamlike-art/dreamlike-photoreal-2.0


