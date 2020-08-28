import os

import numpy as np
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

    pitches in range [0, 11] and durations in range [0, 4] are
    consolidated into single integer according to:
    np.dot([len(dur_types), 1], p_d)
    so that each pitch-duration pair has a unique integer class
    '''

    song_ = np.copy(song) # make a copy of the song so that it isn't modified

    song_[:,1] = map_durs(song_[:, 1])
    song_ = song_.astype(int)

    song_ = np.array([pitch_dur_to_int(p_d) for p_d in song_])

    return song_

# endregion
