# INSTALLAZIONE

## Tramite Compose

## Tramite Dockerfile manualmente

# SISTEMA

I tempi e le considerazioni fatte sono in relazione alle seguenti risorse:

- GPU: NVIDIA GeForce GTX 1060 (6GB GPU dedicated memory)
- RAM: 16 GB
- OS: Windows
- Drive: HDD (no SSD were available)
- CPU Intel Core i5-8600k @ 3.60GHz

## Training LoRA

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

I tempi medi di esecuzione del training nel sistema di riferimento vanno dai 60 ai 90 minuti.

# Resources Link

[1] Kohya: https://github.com/kohya-ss/sd-scripts
