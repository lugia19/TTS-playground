Essentially, this is a re-implementation of OpenAI's API playground with a pyqt frontend.

The actually interesting part is the "Speak" button for each message. 

Essentially, it uses a separate API call to figure out what parts of the text are being spoken by what character, and uses function calling to speak them out loud with the voices defined by the user for each character.