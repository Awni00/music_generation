import os

import numpy as np
import tensorflow as tf
import music21


# region getting note sequencesfrom midi files using music21
def parse_midi_file(filename):
    '''load a midi file as a music21.stream.Score'''

    try:
        return music21.converter.parse(filename)
    except:
        print(f'Error loading {filename}')
        return None

def get_midi_files(directory):

    midi_files = []

    for entry in os.scandir(directory):
        if entry.is_dir():
            midi_files += get_midi_files(entry.path)
        elif entry.path.endswith('.mid'):
            midi_files.append(entry.path)

    return midi_files


def get_pitch(note, chord_root=True, pitch_midi=False):
    '''
    gets the pitch of a note.

    if the note is a chord, either returns the root or a list of pitches.

    pitch is an int in the range [0, 11] representing the pitchClass from 0: C to 11: B.
    if midi: instead of 12-note pitchClass, midi value is returned

    '''

    if pitch_midi:
        if note.isNote:
            return note.pitch.midi
        elif note.isChord:
            if chord_root: return note.findRoot().midi
            else: return [pitch.midi for pitch in note.pitches]
    else:
        if note.isNote:
            return note.pitch.pitchClass
        elif note.isChord:
            if chord_root: return note.findRoot().pitchClass
            else: return note.pitchClasses


def get_notes(mid, chord_root=True, pitch_midi=False, duration_type=True):
    '''
    returns list of tuples of notes as (pitch, duration).

    pitch is an int in the range [0, 11] representing the pitchClass from 0: C to 11: B
    duration is str name of duration or float # of quarter notes

    if chord_root: only root note of chord is returned in pitch
    else: array of pitches

    if midi: instead of 12-note pitchClass, midi value is returned

    if duration_type: str rep of duration returned
    else: float # of quarter nootes
    '''

    notes = mid.recurse().notes

    if duration_type:
        durs = [note.duration.type for note in notes]
        # TODO: duration type is string, so takes lot's of space when dataset is large
        # encode more efficiently
    else:
        durs = [note.duration.quarterLength for note in notes]

    pitch = [get_pitch(note, chord_root=chord_root, pitch_midi=pitch_midi) for note in notes]

    pitch_durs = list(zip(pitch, durs))

    return pitch_durs



def get_pitches(mid, chord_root=True, pitch_midi=False):
    notes = mid.recurse().notes

    pitches = [get_pitch(note, chord_root=chord_root, pitch_midi=pitch_midi) for note in notes]

    return pitches

def get_durs(mid, duration_type=True):

    notes = mid.recurse().notes

    if duration_type:
        durs = [note.duration.type for note in notes]
    else:
        durs = [note.duration.quarterLength for note in notes]

    return durs

# endregion





# region formatting note sequences for generative model

pitches = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
pitch_types = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

duration_types = ['16th', 'eighth', 'quarter', 'half', 'whole', 'zero',
                    'complex', 'breve', 'longa', 'maxima','inexpressible']

note_index = {'C':   0,
              'C#':  1,
              'D':   2,
              'D#':  3,
              'E':   4,
              'F':   5,
              'F#':  6,
              'G':   7,
              'G#':  8,
              'A':   9,
              'A#': 10,
              'B':  11}

inv_note_index = {str(ind): note for note, ind in note_index.items()}

# duration reduction map: map uncommon/unknown durations to standard ones
red_durs_map = {'16th': '16th', 'eighth': 'eighth', 'quarter': 'quarter', 'half': 'half', 'whole': 'whole',
                'zero': 'eighth', 'breve': 'whole', 'longa': 'whole', 'maxima': 'whole',
                'duplex-maxima': 'whole', 'complex': 'quarter', 'inexpressible':'quarter'}
                # the second and third lines are experimental

vec_red_durs = np.vectorize(lambda x: red_durs_map[x])



# duration-integer map: mapping of 4 reduced durations to integers

durs_int_map = {'16th': 0, 'eighth': 1, 'quarter': 2, 'half': 3, 'whole': 4}
inv_durs_int = {i: dur for dur, i in durs_int_map.items()}

vec_durs_int = np.vectorize(lambda x: durs_int_map[x])


def map_durs(durs):
    '''
    returns integer representation of duration types.

    durations are reduced to:
    {'16th': 0, 'eighth': 1, 'quarter': 2, 'half': 3, 'whole': 4}
    with other duration types convereted to closest approximation
    '''

    return vec_durs_int(vec_red_durs(durs))


def pitch_dur_to_int(pitch_dur):
    '''returns integer class of [pitch, duration]'''

    return np.dot([len(durs_int_map.values()), 1], pitch_dur)


def map_song(song):
    '''
    returns integer representation of pitch-duration pairs.

    input:
        song is a 2-D array with column 0: pitches, column 1: durations
        pitches are music21.pitch.pitchClass
        durations are music21.duration.type

    durations are converted to integers using map_durs.

    returns (pitches, durations)
    '''

    song_ = np.copy(song) # make a copy of the song so that it isn't modified

    song_pitches = song_[:, 0].astype(int)
    song_durs = map_durs(song_[:, 1]).astype(int)


    return song_pitches, song_durs

def get_sequences(seq_data, input_length=32, output_length=32, offset=1, shift=1):
    '''
    given a song, returns input and output sequences for training generative model.

    Arguments:
        seq_data: song in the form of a sequence of notes
        input_length: length of input window
        output_length: length of output window
        offset: the offset between the input window and output window
        shift: the shift between consecutive notes

    Returns:
        (X, Y): tuple of input windows and output windows
    '''

    dataset = tf.data.Dataset.from_tensor_slices(seq_data)

    window_length = input_length + offset
    dataset = dataset.window(window_length, shift=shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_length))
    dataset = dataset.map(lambda window: (window[:input_length], window[-output_length:]))

    X = np.array([input for input, label in dataset.as_numpy_iterator()])
    Y = np.array([label for input, label in dataset.as_numpy_iterator()])

    return X, Y

# endregion

# region modeling and music generation from model

def to_ordinal(x):
    '''converts one-hot encoding to ordinal encoding.'''

    return np.argmax(x, axis=-1)

def predict_next(model, pitches, durs):
    '''
    predicts the next note.

    assumes model which takes [pitches, durs] as input and outputs [pitch, dur]
    '''

    model_input = [np.array([pitches]), np.array([durs])]

    pred = model.predict(model_input)
    pred_p, pred_d = pred[0], pred[1]
    pred_p, pred_d= to_ordinal(pred_p), to_ordinal(pred_d)
    pred_p, pred_d = np.squeeze(pred_p), np.squeeze(pred_d)

    return pred_p, pred_d

def predict_next_dist(model, pitches, durs):
    '''
    predicts the probability distribution of the next note.

    assumes model which takes [pitches, durs] as input and outputs [pitch, dur]
    '''

    model_input = [np.array([pitches]), np.array([durs])]

    pred = model.predict(model_input)
    pred_p, pred_d = pred[0], pred[1]
    pred_p, pred_d = np.squeeze(pred_p), np.squeeze(pred_d)


    return pred_p, pred_d

def sample_from_prob(prob_pred, temp=1):
    '''samples a class from a softmax probability logit'''

    prob_pred_ = np.array(prob_pred) ** (1/temp) # apply temperature to softmax logits

    prob_pred_ /= prob_pred_.sum() # re-normalize

    # sample acccording to (adjusted) class probability distribution
    sample = np.random.choice(range(len(prob_pred_)), p=prob_pred_)

    return sample

def generate_sequence(model, pitch_seed, dur_seed, n_notes=64, seed_len=None, temp=1):
    '''generates a sequence of notes.'''

    pitches, durs = [], []

    if not seed_len: seed_len = len(pitch_seed)

    for _ in range(n_notes):
        pitch_seed = pitch_seed[-seed_len:]
        dur_seed = dur_seed[-seed_len:]

        pitch_pred, dur_pred = predict_next_dist(model, pitch_seed, dur_seed)

        pitch = sample_from_prob(pitch_pred, temp=temp)
        dur = sample_from_prob(dur_pred, temp=temp)

        pitches = np.append(pitches, pitch)
        durs = np.append(durs, dur)

        pitch_seed =  np.append(pitch_seed, pitch)
        dur_seed =  np.append(dur_seed, dur)

    return pitches, durs


def generate_stream(pitch_seq, dur_seq, instrument=music21.instrument.Piano()):
    '''generates midi file from sequence of pitches and durations'''

    stream = music21.stream.Stream() # create stream object
    stream.append(instrument) # set instrument

    for pitch, dur in zip(pitch_seq, dur_seq):

        pitch = music21.pitch.Pitch(pitchClass=pitch)
        dur = music21.duration.Duration(type=inv_durs_int[dur])

        stream.append(music21.note.Note(pitch=pitch, duration=dur))

    return stream

def stream_to_midi(stream, filename):
    '''write music21 stream to midi file'''
    stream.write('midi', fp=f'{filename}.mid')

# endregion