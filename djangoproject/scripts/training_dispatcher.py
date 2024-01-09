
import os
import subprocess

def dispatch(lora_id, images):    
    rel_path = "../training-app"
    cwd = os.getcwd()
    training_app_path = os.path.join(cwd, rel_path)
    #(WINDOWS)training_launcher = training_app_path + "/launch.bat"
    training_launcher = training_app_path + "/launch.bash"
    training_input_path = training_app_path + "/training-input/" + lora_id + "/image/300_xkywkrav/"
    os.makedirs(training_input_path)

    for i in range(5):
        text_file = open(training_input_path + str(i+1) + ".txt", "w")
        text_file.write("photo of xkywkrav")
        text_file.close()
        images[i].save(training_input_path + str(i+1) + ".jpg")

    #(WINDOWS)subprocess.Popen([training_launcher, lora_id], cwd=training_app_path, creationflags=subprocess.CREATE_NEW_CONSOLE)
    subprocess.Popen(['sh',training_launcher, lora_id], cwd=training_app_path)