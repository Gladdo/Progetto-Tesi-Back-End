# INSTALLAZIONE

## Tramite Compose

## Tramite Dockerfile manualmente

# PROFILING

I tempi e le considerazioni fatte sono in relazione alle seguenti risorse:

- GPU: NVIDIA GeForce GTX 1060 (6GB GPU dedicated memory)
- RAM: 16 GB
- OS: Windows
- Drive: HDD (no SSD were available)
- CPU Intel Core i5-8600k @ 3.60GHz

## Training LoRA

L'addestramento è eseguito su 5 immagini dell'utente su ciascuna delle quali viene fatto 300 iterazioni di training (per un totale di 1500 iterazioni); queste configurazioni di addestramento sono ritenute minime per quanto riguarda la cultura del fine tuning con LoRA.

L'addestramento è ottenuto facendo uso degli script nel seguente repository [1] e nell'applicazione segue il seguente flow:

- L'utente seleziona nel front-end le 5 immagini con cui creare il modello LoRA e le comunica alla vista LoraTraining nell'app diffuser_api del server Django
- La vista LoraTraining controlla che siano presenti le 5 immagini nel body
- Se il controllo va a buon fine la vista genera un codice di 8 cifre (LoraCode); tale codice è comunicato
-- allo script training_dispatcher che lancia un subprocess dove viene effettuato il training
-- all'utente che poi lo utilizzerà per fare riferimento al proprio modello LoRA


# Resources Link

[1] Kohya: https://github.com/kohya-ss/sd-scripts
