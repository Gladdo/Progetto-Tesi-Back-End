# INSTALLAZIONE

### Tramite Compose

### Tramite Dockerfile manualmente

# SISTEMA

I tempi e le considerazioni fatte sono in relazione alle seguenti risorse:

- GPU: NVIDIA GeForce GTX 1060 (6GB GPU dedicated memory)
- RAM: 16 GB
- OS: Docker Desktop on Windows 10 with WSL2 :( 
- Drive: HDD (no SSD were available)
- CPU Intel Core i5-8600k @ 3.60GHz

L'esecuzione su container Docker richiede che al container venga fornito almeno 9GB di RAM (l'utilizzo medio della RAM è inferiore ma ci sono dei peak sulla richiesta di risorse durante alcune fasi della generazione, come specificato successivamente) 

### Aumentare la RAM fornita ad un container docker ( su windows con WSL2 ):

Nella folder C:\users\<user_name> creare un file .wslconfig con al suo interno le seguenti righe:

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

### Esecuzione dello script di generazione (generator.py)

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

### Alternativa in deploy

### Profiling

Resource peaks: al caricamento dei modelli e alla conversione delle immagini al termine della generazione

## Resources Link

[1] Kohya: https://github.com/kohya-ss/sd-scripts

[2] Diffusers: https://huggingface.co/blog/stable_diffusion

[3] HuggingFace: https://huggingface.co/
