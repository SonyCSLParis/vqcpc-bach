import music21

# constants
SLUR_SYMBOL = '__'
START_SYMBOL = 'START'
END_SYMBOL = 'END'
REST_SYMBOL = 'rest'
OUT_OF_RANGE = 'OOR'
PAD_SYMBOL = 'XX'


def standard_name(note_or_rest, voice_range=None):
    """
    Convert music21 objects to str
    :param note_or_rest:
    :return:
    """
    if isinstance(note_or_rest, music21.note.Note):
        if voice_range is not None:
            min_pitch, max_pitch = voice_range
            pitch = note_or_rest.pitch.midi
            if pitch < min_pitch or pitch > max_pitch:
                return OUT_OF_RANGE
        return note_or_rest.nameWithOctave
    if isinstance(note_or_rest, music21.note.Rest):
        return note_or_rest.name  # == 'rest' := REST_SYMBOL
    if isinstance(note_or_rest, str):
        return note_or_rest

    if isinstance(note_or_rest, music21.harmony.ChordSymbol):
        return note_or_rest.figure
    if isinstance(note_or_rest, music21.expressions.TextExpression):
        return note_or_rest.content


def standard_note(note_or_rest_string):
    if note_or_rest_string == 'rest':
        return music21.note.Rest()
    # treat other additional symbols as rests
    elif note_or_rest_string == END_SYMBOL:
        return music21.note.Note('D~3', quarterLength=1)
    elif note_or_rest_string == START_SYMBOL:
        return music21.note.Note('C~3', quarterLength=1)
    elif note_or_rest_string == PAD_SYMBOL:
        return music21.note.Note('E~3', quarterLength=1)
    elif note_or_rest_string == SLUR_SYMBOL:
        return music21.note.Rest()
    elif note_or_rest_string == OUT_OF_RANGE:
        return music21.note.Rest()
    else:
        return music21.note.Note(note_or_rest_string)