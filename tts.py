import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 175)

engine.say("Hello, this is a local AI speaking.")
engine.runAndWait()
