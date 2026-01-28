import pyttsx3

engine = pyttsx3.init(driverName="espeak")
engine.setProperty("volume", 1.0)
engine.setProperty("rate", 170)

engine.say("If you hear this, text to speech is working")
engine.runAndWait()
