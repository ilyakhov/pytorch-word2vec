import os


def load_bible(path):
    dataset = []
    if os.path.isdir(path):
        path = os.path.join(path, 'bible.txt')
    elif os.path.isfile(path) and path.rsplit('.')[1] != 'txt':
        pass
    else:
        raise NotImplementedError("Didn't support such file extension!")
    with open(path) as f:
        #     dataset = [l for l in f.readlines() if l != '\n']
        buffer = []
        for l in f:
            if len(buffer) > 0 and l[0] in set('0123456789'):
                dataset.append(' '.join(buffer))
                buffer = [l]
            elif len(buffer) == 0:
                buffer.append(l)
            elif l[0] not in set('0123456789'):
                buffer.append(l)
        if len(buffer) > 0:
            dataset.append(' '.join(buffer))
    return dataset


def split_bible_on_chapters(dataset):
    chapters = []
    current_chapter = []
    prev_chapter_number = None
    for sentence in dataset:
        chapter_number = sentence[0]
        if prev_chapter_number is None:
            current_chapter.append(sentence)
            prev_chapter_number = chapter_number
        elif prev_chapter_number == chapter_number:
            current_chapter.append(sentence)
        else:
            try:
                chapter_number = int(chapter_number)
            except:
                pass
            if isinstance(chapter_number, int):
                current_chapter = ' '.join(current_chapter)
                chapters.append(current_chapter)
                current_chapter = [sentence]
                prev_chapter_number = str(chapter_number)
            else:
                current_chapter.append(sentence)
                current_chapter = ' '.join(current_chapter)
                chapters.append(current_chapter)
                current_chapter = []
                prev_chapter_number = None

    if len(current_chapter) > 0:
        chapters.extend(current_chapter)

    return chapters