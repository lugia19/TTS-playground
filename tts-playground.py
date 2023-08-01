import copy
import hashlib
import json
import os
import queue
import re
import threading
import time

import elevenlabslib.helpers
import requests
from PyQt6.QtGui import QTextLayout, QTextOption, QFont, QKeySequence, QShortcut
from elevenlabslib import *
import keyring
import openai
from PyQt6 import QtGui, QtWidgets, QtCore
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QScrollArea, QComboBox, QTextEdit, QHBoxLayout, QLabel, QDialog, QCheckBox, QSpacerItem, QSizePolicy, QFileDialog, \
    QMessageBox
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QThread
from PyQt6.QtWidgets import QGridLayout

settings = dict()

colors_dict = {
    "primary_color":"#1A1D22",
    "secondary_color":"#282C34",
    "hover_color":"#596273",
    "text_color":"#FFFFFF",
    "toggle_color":"#4a708b",
    "green":"#3a7a3a",
    "yellow":"#faf20c",
    "red":"#7a3a3a"
}
synthetizer = None
cachedVoices = dict()

def get_stylesheet():
    with open("stylesheet.qss", "r", encoding="utf8") as fp:
        styleSheet = fp.read()

    for colorKey, colorValue in colors_dict.items():
        styleSheet = styleSheet.replace("{" + colorKey + "}", colorValue)
    return styleSheet

def reload_settings():
    global settings
    if os.path.exists("config.json"):
        with open("config.json", "r", encoding="utf8") as fp:
            settings = json.load(fp)
    else:
        settings = dict()
        with open("config.json", "w", encoding="utf8") as fp:
            json.dump(settings, fp, indent=4)

def dump_settings():
    with open("config.json", "w", encoding="utf8") as fp:
        json.dump(settings, fp, indent=4)

reload_settings()


#This is the function used for function calling
def speak(speechInfo:str, characterAssociations:dict, text_hash:str):
    speechInfo = json.loads(speechInfo)

    characterAssociationsLowercase = {character.lower(): voiceName for character, voiceName in characterAssociations.items()}

    for speechEntry in speechInfo:
        character = speechEntry["character"]
        prompt = re.sub(r'\.(?=\s)', '...', speechEntry["prompt"]).strip()
        if prompt.endswith(","):
            prompt = prompt[:-1] + '.'

        voiceName = characterAssociationsLowercase.get(character.lower())
        if voiceName is not None and voiceName.lower() != "none":
            global synthetizer
            if synthetizer is None:
                synthetizer = Synthesizer(keyring.get_password("playground","elevenlabs_api_key"))
                threading.Thread(target=synthetizer.main_loop).start()
            if voiceName not in cachedVoices:
                cachedVoices[voiceName] = synthetizer.user.get_voices_by_name(voiceName)[0]
            synthetizer.ttsQueue.put((prompt, cachedVoices[voiceName], character, text_hash))



#The Synthesizer class is just borrowed from another project I'm working on, with some tweaks to make it save the generated audio files.
class Synthesizer:
    def __init__(self, apiKey:str):
        self.eventQueue = queue.Queue()
        self.readyForPlaybackEvent = threading.Event()
        self.readyForPlaybackEvent.set()
        self.user = elevenlabslib.ElevenLabsUser(apiKey)

        self.ttsQueue = queue.Queue()
        self.interruptEvent = threading.Event()
        self.isRunning = threading.Event()
        self.isRunning.set()

    def main_loop(self):
        threading.Thread(target=self.waitForPlaybackReady).start()  # Starts the thread that handles playback ordering.
        while True:
            try:
                prompt, voice, character, texthash = self.ttsQueue.get(timeout=5)
            except queue.Empty:
                if self.interruptEvent.is_set():
                    print("Synthetizer main loop exiting...")
                    return
                continue
            if self.isRunning.is_set():
                print(f"{character}: {prompt}")
                self.synthesizeAndPlayAudio(prompt, voice, texthash)

    def synthesizeAndPlayAudio(self, prompt, voice:ElevenLabsVoice, text_hash) -> None:
        newEvent = threading.Event()
        self.eventQueue.put(newEvent)
        def startcallbackfunc():
            newEvent.wait()
        def endcallbackfunc():
            self.readyForPlaybackEvent.set()
            audio_folder = os.path.join('audio', text_hash)
            if not os.path.exists(audio_folder):
                os.mkdir(audio_folder)

            #Let's make sure it's done...
            time.sleep(5)
            #We look through the last 10 generated items to see if we can find it.
            try:
                historyItemDict = self.user.download_history_items_v2(self.user.get_history_items_paginated(10))
            except requests.HTTPError:
                return

            historyItem = None
            downloadedData = None
            for item, downloadedData in historyItemDict.items():
                if item.text != prompt:
                    continue
                else:
                    downloadedData = downloadedData
                    historyItem = item
                    break
            if historyItem is None or downloadedData is None:
                return

            print(downloadedData[1])
            with open(os.path.join(audio_folder,downloadedData[1]), "wb") as fp:
                fp.write(downloadedData[0])
                fp.flush()


        playbackOptions = PlaybackOptions(runInBackground=True, onPlaybackStart=startcallbackfunc, onPlaybackEnd=endcallbackfunc)
        generationOptions = GenerationOptions(latencyOptimizationLevel=3,
                                              model_id=settings["tts_model"],
                                              use_speaker_boost=settings["speaker_boost"],
                                              stability=float(settings["stability"]),
                                              similarity_boost=float(settings["similarity"]),
                                              style=float(settings["style"]))
        voice.generate_stream_audio_v2(prompt=prompt, playbackOptions=playbackOptions, generationOptions=generationOptions)
    def waitForPlaybackReady(self):
        while True:
            self.readyForPlaybackEvent.wait()
            self.readyForPlaybackEvent.clear()
            while True:
                try:
                    nextEvent = self.eventQueue.get(timeout=5)
                except queue.Empty:
                    if self.interruptEvent.is_set():
                        print("Synthetizer playback loop exiting...")
                        return
                    continue
                time.sleep(2)
                nextEvent.set()
                break



#Used for the auto-resizing of the text fields
def visibleLineCount(textEdit):
    text_blocks = textEdit.toPlainText().split('\n')
    option = QTextOption()
    option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
    leading = textEdit.fontMetrics().leading()
    width = textEdit.width() - textEdit.document().documentMargin()*2

    total_lines = 0
    for block in text_blocks:
        layout = QTextLayout(block, textEdit.font())
        layout.setTextOption(option)
        layout.beginLayout()

        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(width)
            total_lines += 1

        layout.endLayout()

    return total_lines

class CenteredLabel(QtWidgets.QLabel):
    def __init__(self, text=None, wordWrap=False):
        super(CenteredLabel, self).__init__(text)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(wordWrap)
class LabeledInput(QtWidgets.QWidget):
    """
    This widget has a label and below it an input.
    Arguments:
        label: The text to put above the input
        configKey: The corresponding configKey to pull the default value from and save the user-selected value to
        data: The options to choose from. If it's a string, the input will be a lineEdit, if it's a list, it will be a comboBox.
        protected: Saves the config data to the system keyring instead of the 'settings' dict.
    """
    def __init__(self, label, configKey, data=None, protected=False):
        super().__init__()
        self.configKey = configKey
        self.layout = QtWidgets.QVBoxLayout(self)
        self.label = CenteredLabel(label)
        self.layout.addWidget(self.label)
        self.protected = protected
        self.line_edit = None
        self.combo_box = None

        self.input_widget = QtWidgets.QWidget()
        self.input_layout = QtWidgets.QHBoxLayout(self.input_widget)
        self.input_layout.setSpacing(10)  # adjust the space between widgets

        if isinstance(data, list):
            self.combo_box = QtWidgets.QComboBox()
            self.combo_box.addItems(data)
            self.input_layout.addWidget(self.combo_box)
        else:
            self.line_edit = QtWidgets.QLineEdit()
            self.line_edit.setText(data)
            if protected:
                self.line_edit.setEchoMode(QtWidgets.QLineEdit.EchoMode.PasswordEchoOnEdit)
            self.input_layout.addWidget(self.line_edit)

        currentValue = None
        if protected:
            currentValue = keyring.get_password("playground", configKey)
        else:
            if configKey in settings:
                currentValue = settings[configKey]

        if currentValue is not None:
            if isinstance(data, list):
                allItems = [self.combo_box.itemText(i) for i in range(self.combo_box.count())]
                if currentValue in allItems:
                    self.combo_box.setCurrentIndex(allItems.index(currentValue))
                else:
                    self.combo_box.setCurrentIndex(0)
            else:
                self.line_edit.setText(str(currentValue))

        self.layout.addWidget(self.input_widget)

    def get_value(self):
        if self.line_edit is not None:
            return self.line_edit.text()
        else:
            return self.combo_box.currentText()
class ConfigCheckBox(QCheckBox):
    def __init__(self, label, configKey, parent=None):
        super(ConfigCheckBox, self).__init__(label, parent)
        self.configKey = configKey

    def get_value(self):
        return self.isChecked()
class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super(SettingsDialog, self).__init__(parent)

        self.layout = QGridLayout()

        # Model settings
        openAIKey = keyring.get_password("playground","openai_api_key")
        modelIDs = "OpenAI key missing"
        try:
            if openAIKey is not None:
                modelIDs = list()
                openai.api_key = openAIKey
                allModels = openai.Model.list()
                for model in allModels["data"]:
                    if "gpt-4" in model.id or "gpt-3.5" in model.id:
                        modelIDs.append(model.id)
        except openai.error.AuthenticationError:
            pass

        self.chat_model = LabeledInput("Chat Model", "openai_model", data=modelIDs)
        if isinstance(modelIDs, str):
            self.chat_model.line_edit.setText(modelIDs)
            self.chat_model.line_edit.setDisabled(True)

        self.layout.addWidget(self.chat_model, 0, 0)

        self.temperature = LabeledInput("Temperature", "temperature", data="1.0")
        self.layout.addWidget(self.temperature, 1, 0)

        self.maxLength = LabeledInput("Maximum Length", "max_tokens", data="256")
        self.layout.addWidget(self.maxLength, 2, 0)

        self.topP = LabeledInput("Top P", "top_p", data="0.7")
        self.layout.addWidget(self.topP, 3, 0)

        self.frequencyPenalty = LabeledInput("Frequency Penalty", "frequency_penalty", data="0.3")
        self.layout.addWidget(self.frequencyPenalty, 4, 0)

        # TTS settings
        elevenlabsKey = keyring.get_password("playground","elevenlabs_api_key")

        modelIDs = "ElevenLabs key missing"
        try:
            if elevenlabsKey is not None:
                modelsList = ElevenLabsUser(elevenlabsKey).get_available_models()
                modelIDs = list()
                for model in modelsList:
                    modelIDs.append(model["model_id"])
        except ValueError:
            pass

        self.tts_model = LabeledInput("TTS Model", "tts_model", modelIDs)
        if isinstance(modelIDs, str):
            self.tts_model.line_edit.setText(modelIDs)
            self.tts_model.line_edit.setDisabled(True)

        self.layout.addWidget(self.tts_model, 0, 1)

        self.stability = LabeledInput("Stability", "stability", data="0.4")
        self.layout.addWidget(self.stability, 1, 1)

        self.similarity = LabeledInput("Similarity", "similarity", data="0.9")
        self.layout.addWidget(self.similarity, 2, 1)

        self.style = LabeledInput("Style", "style", data="0.0")
        self.layout.addWidget(self.style, 3, 1)

        self.speakerBoost = ConfigCheckBox("Speaker Boost", "speaker_boost")
        if "speaker_boost" in settings:
            self.speakerBoost.setChecked(settings["speaker_boost"])
        else:
            self.speakerBoost.setChecked(True)
        self.layout.addWidget(self.speakerBoost, 4, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        self.openai_api = LabeledInput("OpenAI API Key", "openai_api_key", protected=True)
        self.layout.addWidget(self.openai_api, 5, 0)

        self.eleven_api = LabeledInput("ElevenLabs API Key", "elevenlabs_api_key", protected=True)
        self.layout.addWidget(self.eleven_api, 5, 1)

        self.okButton = QPushButton("Save")
        self.okButton.clicked.connect(self.save)  # This will close the dialog
        self.layout.addWidget(self.okButton, 6, 0, 1, 2)  # Span both columns

        self.setLayout(self.layout)

    def iterate_widgets(self,layout):
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                yield widget
                if isinstance(widget, QtWidgets.QWidget):
                    if callable(widget.layout):
                        child_layout = widget.layout()
                    else:
                        child_layout = widget.layout
                    if child_layout is not None:
                        yield from self.iterate_widgets(child_layout)
            else:
                child_layout = layout.itemAt(i).layout()
                if child_layout is not None:
                    yield from self.iterate_widgets(child_layout)
    def save(self):
        newSettings = copy.deepcopy(settings)

        for widget in self.iterate_widgets(self.layout):
            if hasattr(widget, 'configKey'):
                #Read and save the config data.
                configKey = widget.configKey
                value = widget.get_value()

                useKeyring = hasattr(widget, "protected") and widget.protected

                if useKeyring:
                    configKey = "keyring_" + configKey

                if value is not None:
                    newSettings[configKey] = value

        keysToPop = list()
        for key, value in newSettings.items():
            if "keyring_" in key:
                keyringKey = key[len("keyring_"):]
                keyring.set_password("playground", keyringKey, value)
                keysToPop.append(key)
        for key in keysToPop:
            newSettings.pop(key)
        for key, value in newSettings.items():
            settings[key] = value
        dump_settings()
        self.accept()
class GrowingTextEdit(QTextEdit):
    sizeChanged = pyqtSignal(int)

    def __init__(self, *args, **kwargs):
        super(GrowingTextEdit, self).__init__(*args, **kwargs)
        font = QFont()
        font.setPointSize(12)  # Set the font size to 12
        self.setFont(font)

        self.document().contentsChanged.connect(self.scheduleSizeChange)
        self.setMinimumHeight(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.heightMin = 0
        self.heightMax = 65000
        self.currentHeight = self.document().size().height()

    def scheduleSizeChange(self):
        QTimer.singleShot(0, self.sizeChange)

    def sizeChange(self):
        numLines = visibleLineCount(self)
        metrics = self.fontMetrics()
        lineHeight = metrics.lineSpacing()

        newHeight = int(numLines * lineHeight + 15)
        if self.heightMin <= newHeight <= self.heightMax:
            delta = newHeight - self.currentHeight
            self.setMinimumHeight(newHeight)
            self.currentHeight = newHeight
            self.sizeChanged.emit(delta)
class Message(QWidget):
    def __init__(self, parent=None, text=None):
        super(Message, self).__init__(parent)

        self.layout = QHBoxLayout()
        self.setMinimumHeight(125)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.comboBox = QComboBox()
        self.comboBox.addItems(["User", "Assistant"])
        self.comboBox.currentTextChanged.connect(self.checkUserRole)
        self.layout.addWidget(self.comboBox)

        self.textBox = GrowingTextEdit()
        self.textBox.sizeChanged.connect(self.adjustSizeCustom)

        self.layout.addWidget(self.textBox)

        self.speakButton = QPushButton("Speak")
        self.speakButton.clicked.connect(self.speakClicked)
        self.layout.addWidget(self.speakButton)

        self.deleteButton = QPushButton("Delete")
        self.deleteButton.clicked.connect(self.delete)
        self.layout.addWidget(self.deleteButton)

        self.setLayout(self.layout)

        self.checkUserRole()  # Check initially when the widget is created

        if text is not None:
            self.textBox.setText(text)
        QTimer.singleShot(25, lambda: self.adjustSizeCustom(0))

    def delete(self):
        self.setParent(None)

    def connectSizeChanged(self):
        chatInterface = self

        while chatInterface is not None and not isinstance(chatInterface, ChatInterface):
            chatInterface = chatInterface.parent()

        if chatInterface is not None:
            self.textBox.sizeChanged.connect(chatInterface.scrollBy)

    def adjustSizeCustom(self, delta) -> None:
        numLines = visibleLineCount(self.textBox)
        metrics = self.textBox.fontMetrics()
        lineHeight = metrics.lineSpacing()

        textBoxHeight = numLines * lineHeight
        self.setFixedHeight(int(textBoxHeight)+30)

    def speakClicked(self):
        if keyring.get_password("playground", "openai_api_key") is None:
            # Show a message box
            QMessageBox.warning(self, "API Key Missing", "The OpenAI API key was not found. Please add it in the settings.")
            return

        if keyring.get_password("playground", "elevenlabs_api_key") is None:
            # Show a message box
            QMessageBox.warning(self, "API Key Missing", "The ElevenLabs API key was not found. Please add it in the settings.")
            return

        if "key missing" in settings["tts_model"].lower():
            QMessageBox.warning(self, "Model not set", "ElevenLabs TTS model not set. Please set it in the settings.")
            return

        text = self.textBox.toPlainText()
        self.speakButton.setDisabled(True)
        # Calculate the hash of the current text
        hash_object = hashlib.md5(text.encode())
        text_hash = hash_object.hexdigest()
        if not os.path.exists("audio"):
            os.mkdir("audio")

        # Check if the audio has already been generated
        audio_folder = os.path.join('audio', text_hash)
        if os.path.exists(audio_folder):
            # Get the corresponding audio files
            audio_files = sorted(os.listdir(audio_folder))
            def wrapperFunc():
                for audio_file in audio_files:
                    # Play the audio file
                    elevenlabslib.helpers.play_audio_bytes_v2(open(os.path.join(audio_folder, audio_file),"rb").read(), PlaybackOptions(runInBackground=False))
                    # Wait for 2 seconds
                    time.sleep(2)
            threading.Thread(target=wrapperFunc).start()
        else:
            # Prompt the API to extract the dialog, then generate it
            messages = list()
            functions = list()
            openai.api_key = keyring.get_password("playground","openai_api_key")
            functions.append(
                {
                    "name": "speak",
                    "description": "Speaks the given prompt as a chosen character.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "speechInfo": {
                                "type": "string",
                                "description": "A JSON array of objects. Each object should have two keys, 'character' and 'prompt', containing the character name and the text to speak respectively."
                            }
                        },
                        "required": ["speechInfo"]
                    },
                }
            )

            chatInterfaceParent = self

            while chatInterfaceParent is not None and not isinstance(chatInterfaceParent, ChatInterface):
                chatInterfaceParent = chatInterfaceParent.parent()


            messages.append({"role": "system", "content": "Your job is to act as a dialog speaker. You will extract the quoted (in quotes) dialog from the provided input and speak it using the provided function.\nYou must NOT modify the text, only extract the dialog.\nIf a character does not has an associated voice, ignore it."})
            characterAssociations = dict()
            if chatInterfaceParent is not None:
                actorsText = chatInterfaceParent.actorsMessage.textBox.toPlainText()
                actorsList = actorsText.split("\n")

                for actorLine in actorsList[1:]:
                    parts = actorLine.replace('\"', "").replace("\'", "").strip().split(":")
                    if len(parts) != 2:
                        continue
                    characterAssociations[parts[0].strip()] = parts[1].strip()

                messages[-1]["content"] += f"\nThe characters you can choose from are: {', '.join(characterAssociations.keys())}"
            messages.append({"role":"user", "content": self.textBox.toPlainText()})
            messages.append({"role": "user", "content": "Please speak the dialog from the previous message."})

            def wrapperFunc(message):
                if message.get("function_call"):
                    function_name = message["function_call"]["name"]
                    argsString = message.get("function_call").get("arguments")
                    argsDict = json.loads(argsString)
                    print(argsDict)
                    if function_name == "speak":
                        if chatInterfaceParent is not None:
                            speak(argsDict.get("speechInfo"), characterAssociations, text_hash)
                    print("Done calling the function")
                self.speakButton.setDisabled(False)

            apiThread = OpenAIAPIThread(messages, functions, functions[0])
            apiThread.finished.connect(wrapperFunc)
            apiThread.start()

    def checkUserRole(self):
        if self.comboBox.currentText() == "User":
            self.speakButton.setDisabled(True)
        else:
            self.speakButton.setDisabled(False)

    def getMessage(self):
        return {
            'role': self.comboBox.currentText(),
            'content': self.textBox.toPlainText()
        }


#QThread that handles calling the openAI API.
class OpenAIAPIThread(QThread):
    finished = pyqtSignal(object)

    def __init__(self, messages, functions=None, function_call=None):
        super().__init__()
        self.messages = messages
        self.functions = functions
        self.function_call = function_call

    def run(self):
        openai.api_key = keyring.get_password("playground","openai_api_key")
        if self.functions is None:
            response = openai.ChatCompletion.create(
                model=settings["openai_model"],
                messages=self.messages,
                top_p=float(settings["top_p"]),
                temperature=float(settings["temperature"]),
                presence_penalty=0.0,
                frequency_penalty=float(settings["frequency_penalty"]),
                max_tokens=int(settings["max_tokens"])
            )
            print(response['choices'][0]['message']['content'])
        else:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=self.messages,
                functions=self.functions,
                function_call=self.function_call
            )
            print(response['choices'][0]['message']['content'])

        self.finished.emit(response['choices'][0]['message'])
class ChatInterface(QWidget):
    def __init__(self, parent=None):
        super(ChatInterface, self).__init__(parent)

        self.settingsDialog = None
        self.layout = QGridLayout()

        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.scrollAreaContent = QWidget()
        self.scrollLayout = QVBoxLayout(self.scrollAreaContent)
        self.scrollArea.setWidget(self.scrollAreaContent)

        self.actorsMessage = Message()
        self.actorsMessage.speakButton.setParent(None)
        self.actorsMessage.deleteButton.setParent(None)
        self.actorsMessage.comboBox.addItem("Actors")
        self.actorsMessage.comboBox.setCurrentText("Actors")
        self.actorsMessage.comboBox.setDisabled(True)
        self.actorsMessage.textBox.setText("Add character-voice pairs here by making new lines in the format 'CharacterName: Voice Name'. If there's a character you do not wish to have a voice, put the Voice Name as 'None'")
        self.actorsMessage.setMinimumHeight(75)
        self.scrollLayout.addWidget(self.actorsMessage)

        self.systemMessage = Message()
        self.systemMessage.comboBox.addItem("System")
        self.systemMessage.comboBox.setCurrentText("System")
        self.systemMessage.comboBox.setDisabled(True)
        self.systemMessage.speakButton.setParent(None)
        self.systemMessage.deleteButton.setParent(None)
        self.scrollLayout.addWidget(self.systemMessage)


        self.layout.addWidget(self.scrollArea, 0, 0, 1, 6)

        self.settingsButton = QPushButton("Settings")
        self.settingsButton.clicked.connect(self.openSettings)
        self.layout.addWidget(self.settingsButton, 1, 0)

        self.exportButton = QPushButton("Export")
        self.exportButton.clicked.connect(self.exportMessagesClicked)
        self.layout.addWidget(self.exportButton, 1, 1)

        self.importButton = QPushButton("Import")
        self.importButton.clicked.connect(self.importMessagesClicked)
        self.layout.addWidget(self.importButton, 1, 2)

        self.layout.addItem(QSpacerItem(20, 25, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum), 1, 3)

        self.addButton = QPushButton("Add")
        self.addButton.clicked.connect(self.addMessage)
        self.layout.addWidget(self.addButton, 1, 4)

        self.sendButton = QPushButton("Send")
        self.sendButton.clicked.connect(self.sendButtonClicked)
        self.layout.addWidget(self.sendButton, 1, 5)

        self.setLayout(self.layout)

        # Create a new QShortcut for the key combination Ctrl+Return
        self.sendShortcutReturn = QShortcut(QKeySequence("Ctrl+Return"), self)
        # Connect the activated signal of the shortcut to the sendMessage method
        self.sendShortcutReturn.activated.connect(self.sendButtonClicked)

        # Create a new QShortcut for the key combination Ctrl+Enter (on the numpad)
        self.sendShortcutEnter = QShortcut(QKeySequence("Ctrl+Enter"), self)
        # Connect the activated signal of the shortcut to the sendMessage method
        self.sendShortcutEnter.activated.connect(self.sendButtonClicked)

        self.actorsMessage.connectSizeChanged()
        self.systemMessage.connectSizeChanged()

        # Create a QTimer instance
        self.resizeTimer = QTimer(self)
        # Set it as single-shot
        self.resizeTimer.setSingleShot(True)
        # Connect the timer timeout signal to your function
        self.resizeTimer.timeout.connect(lambda: self.importMessagesFromList(self.exportMessagesToList()))

        self.scrollLayout.addStretch()

    def scrollBy(self, delta):
        if delta <= 0:
            return
        try:
            self.scrollArea.verticalScrollBar().setValue(int(self.scrollArea.verticalScrollBar().value() + delta * 11 / 10 + 5))
        except OverflowError:
            self.scrollArea.verticalScrollBar().setValue(int(self.scrollArea.verticalScrollBar().maximum()))

    def resizeEvent(self, event):
        super().resizeEvent(event)  # Let the parent class handle the resizing
        self.resizeTimer.start(100)

    def iterate_widgets(self,layout):
        for i in range(layout.count()):
            widget = layout.itemAt(i).widget()
            if widget is not None:
                yield widget
                if isinstance(widget, QtWidgets.QWidget):
                    if callable(widget.layout):
                        child_layout = widget.layout()
                    else:
                        child_layout = widget.layout
                    if child_layout is not None:
                        yield from self.iterate_widgets(child_layout)
            else:
                child_layout = layout.itemAt(i).layout()
                if child_layout is not None:
                    yield from self.iterate_widgets(child_layout)

    def clearAllMessages(self):
        for i in reversed(range(self.scrollLayout.count())):
            item = self.scrollLayout.takeAt(i)  # takeAt removes the item from the layout
            widget = item.widget()
            if widget is not None:  # Check if the item is a widget
                # Destroy the widget
                widget.setParent(None)

    def importMessagesClicked(self):
        # Ask the user where to load the file from
        save_path = os.path.join(os.getcwd(), 'saves')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fileName, _ = QFileDialog.getOpenFileName(self, "Import File", save_path, "JSON Files (*.json)")
        if fileName:
            with open(fileName, 'r') as f:
                messages = json.load(f)



            # Check if it's an openai copy:
            if isinstance(messages, dict):
                messages:list = messages["messages"]
                #Let's check if there is a system message:
                hasSystem = False
                for message in messages:
                    if message["role"].lower() == "system":
                        hasSystem = True
                if not hasSystem:
                    messages.insert(0, {
                        "role": "System",
                        "content": ""
                    })

                #This means the voice message will be missing.
                messages.insert(0, {
                    "role":"Actors",
                    "content":"Add character-voice pairs here by making new lines in the format 'CharacterName: Voice Name'. If there's a character you do not wish to have a voice, put the Voice Name as 'None'"
                })

            self.importMessagesFromList(messages)

    def importMessagesFromList(self, messages):
        # Clear all current messages
        self.clearAllMessages()

        # Add the messages to the chat interface
        for message in messages:
            message["role"] = message["role"][0].upper() + message["role"][1:]
            newMessage = Message()
            if message["role"].lower() == "system":
                self.systemMessage = newMessage
                self.systemMessage.comboBox.addItem("System")
                self.systemMessage.comboBox.setCurrentText("System")
                self.systemMessage.comboBox.setDisabled(True)
                self.systemMessage.speakButton.setParent(None)
                self.systemMessage.deleteButton.setParent(None)
            elif message["role"].lower() == "actors":
                self.actorsMessage = newMessage
                self.actorsMessage.comboBox.addItem("Actors")
                self.actorsMessage.comboBox.setCurrentText("Actors")
                self.actorsMessage.comboBox.setDisabled(True)
                self.actorsMessage.speakButton.setParent(None)
                self.actorsMessage.deleteButton.setParent(None)
                self.actorsMessage.setMinimumHeight(75)
            else:
                newMessage.comboBox.setCurrentText(message['role'])

            self.scrollLayout.addWidget(newMessage)
            newMessage.connectSizeChanged()
            newMessage.textBox.setText(message['content'])
            # newMessage.adjustSize()
        QTimer.singleShot(50, self.scrollToBottom)
        self.scrollLayout.addStretch()
    def scrollToBottom(self):
        self.scrollArea.verticalScrollBar().setValue(self.scrollArea.verticalScrollBar().maximum())

    def exportMessagesToList(self) -> list:
        messages = list()
        for i in range(self.scrollLayout.count()):
            widget = self.scrollLayout.itemAt(i).widget()
            if hasattr(widget, "getMessage"):
                messages.append(widget.getMessage())
        return messages
    def exportMessagesClicked(self):
        # Get all previous messages
        messages = self.exportMessagesToList()

        # Ask the user where to save the file
        save_path = os.path.join(os.getcwd(), 'saves')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fileName, _ = QFileDialog.getSaveFileName(self, "Export File", save_path, "JSON Files (*.json)")
        if fileName:
            with open(fileName, 'w') as f:
                json.dump(messages, f, indent=4)

    def openSettings(self):
        self.settingsDialog = SettingsDialog(self)
        self.settingsDialog.exec()

    def addMessage(self, text:str=None, sender=None):
        self.scrollLayout.takeAt(self.scrollLayout.count() - 1)

        if isinstance(text, bool):
            text = None
        newMessage = Message(text=text)

        if sender is None:
            # Check the sender of the latest message
            latest_message = self.scrollLayout.itemAt(self.scrollLayout.count() - 1).widget()
            if hasattr(latest_message, "comboBox"):
                if latest_message.comboBox.currentText() in {"System", "Assistant"}:
                    newMessage.comboBox.setCurrentText("User")
                elif latest_message.comboBox.currentText() == "User":
                    newMessage.comboBox.setCurrentText("Assistant")
        else:
            newMessage.comboBox.setCurrentText(sender)

        self.scrollLayout.addWidget(newMessage)

        self.scrollLayout.addStretch()

        newMessage.connectSizeChanged()
        QTimer.singleShot(50, self.scrollToBottom)
    def sendButtonClicked(self):
        if keyring.get_password("playground", "openai_api_key") is None:
            # Show a message box
            QMessageBox.warning(self, "API Key Missing", "The OpenAI API key was not found. Please add it in the settings.")
            return

        if "key missing" in settings["openai_model"].lower():
            QMessageBox.warning(self, "Model not set", "OpenAI model not set. Please set it in the settings.")
            return


        # Get all previous messages and do something
        messages = list()
        for widget in self.iterate_widgets(self.scrollLayout):
            if hasattr(widget, "getMessage"):
                message = widget.getMessage()
                message["role"] = message["role"].lower()
                if message["role"] == "actors":
                    continue
                messages.append(message)
        self.sendButton.setDisabled(True)
        self.addButton.setDisabled(True)
        apiThread = OpenAIAPIThread(messages)
        apiThread.finished.connect(self.addAssistantMessage)
        apiThread.start()

    def addAssistantMessage(self, message):
        self.sendButton.setDisabled(False)
        self.addButton.setDisabled(False)
        self.addMessage(message['content'], "Assistant")

if __name__ == "__main__":
    app = QApplication([])
    app.setStyleSheet(get_stylesheet())
    widget = ChatInterface()
    widget.show()
    app.exec()
