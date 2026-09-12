# Music Generation with AI - CodeAlpha Task 3
# By Barla Kavya

import numpy as np
import os
from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# 1. COLLECT MIDI DATA
def get_notes():
    notes = []
    # If you have MIDI files folder, put path here
    # For demo, we use sample notes (C, D, E, F, G scale)
    # In real project, parse MIDI files using music21
    print("Collecting MIDI data...")

    # Example: Parse all midi files in data folder
    # Uncomment when you have MIDI files
    """
    for file in os.listdir(\"midi_data\"):
        if file.endswith(\".mid\"):
            midi = converter.parse(f\"midi_data/{file}\")
            parts = instrument.partitionByInstrument(midi)
            if parts:
                notes_to_parse = parts.parts[0].recurse()
            else:
                notes_to_parse = midi.flat.notes
            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
    """

    # For submission, we create a sample sequence (C major scale pattern)
    sample_notes = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5'] * 50
    sample_notes += ['E4', 'G4', 'C5', 'E5', 'G4', 'E4'] * 20
    return sample_notes

# 2. PREPROCESS DATA
def prepare_sequences(notes, n_vocab):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 20
    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in seq_in])
        network_output.append(note_to_int[seq_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    network_output = to_categorical(network_output)

    return network_input, network_output, pitchnames, note_to_int

# 3. BUILD MODEL - LSTM
def create_model(network_input, n_vocab):
    model = Sequential([
        LSTM(256, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True),
        Dropout(0.3),
        LSTM(256),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(n_vocab, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# 4. TRAIN & 5. GENERATE MIDI
def generate_notes(model, network_input, pitchnames, note_to_int):
    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))
    start = np.random.randint(0, len(network_input)-1)
    pattern = network_input[start]
    prediction_output = []

    # generate 100 notes
    for _ in range(100):
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(prediction_input, verbose=0)
        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)
        pattern = np.append(pattern, [[index / float(len(pitchnames))]], axis=0)
        pattern = pattern[1:len(pattern)]

    return prediction_output

def create_midi(prediction_output, filename='generated_music.mid'):
    offset = 0
    output_notes = []
    for pattern in prediction_output:
        if '.' in pattern: # chord
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else: # single note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)
    print(f"Music saved as {filename}")

# MAIN FLOW
if __name__ == "__main__":
    notes = get_notes()
    n_vocab = len(set(notes))
    print(f"Total Notes: {len(notes)}, Vocab: {n_vocab}")

    network_input, network_output, pitchnames, note_to_int = prepare_sequences(notes, n_vocab)

    model = create_model(network_input, n_vocab)
    print(model.summary())

    # Training - 5 epochs for demo (increase to 50+ for better results)
    model.fit(network_input, network_output, epochs=5, batch_size=64, verbose=1)

    # Generate
    predicted = generate_notes(model, network_input, pitchnames, note_to_int)
    create_midi(predicted)
    print("🎵 Music Generation Complete!")
