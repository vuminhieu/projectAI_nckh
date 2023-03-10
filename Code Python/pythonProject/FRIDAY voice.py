import pyttsx3
import datetime

friday = pyttsx3.init()
voice = friday.getProperty('voices')
friday.setProperty('voice', voice[1].id)

def speak(audio) :
    friday.say(audio)
    friday.runAndWait()

def time() :
    Time = datetime.datetime.now().strftime('%I:%M:%p')
    speak(Time)
    print(Time)

time()
speak('Hello Trung , I\'m Friday ')