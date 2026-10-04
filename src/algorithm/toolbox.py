"""
Toolbox and Logbook - DEAP-style operator registry and evolution recorder.

Toolbox   - Stores and partially-applies evolutionary operators under named
            aliases (clone, map, evaluate, generate, update, ...).
Logbook   - A chronological list of dicts recording per-generation statistics,
            plus a free-form 'bookshelf' dict for arbitrary run metadata.
"""

from copy import deepcopy
from functools import partial
from collections import defaultdict
from itertools import chain


class Toolbox(object):
    """A toolbox for evolution that contains the evolutionary operators.

    At construction the toolbox registers two default operators:
        clone  - defaults to copy.deepcopy
        map    - defaults to the built-in map

    Additional operators are added via :meth:`register`.
    """

    def __init__(self):
        self.register("clone", deepcopy)
        self.register("map", map)
        self.logbook = Logbook()

    def register(self, alias, function, *args, **kargs):
        """Register *function* under *alias*, optionally pre-binding arguments.

        After registration, ``toolbox.<alias>(...)`` calls
        ``function(*args, **kargs, ...)`` with the pre-bound arguments
        prepended/merged at call time.
        """
        pfunc = partial(function, *args, **kargs)
        pfunc.__name__ = alias
        pfunc.__doc__ = function.__doc__

        if hasattr(function, "__dict__") and not isinstance(function, type):
            pfunc.__dict__.update(function.__dict__.copy())

        setattr(self, alias, pfunc)

    def unregister(self, alias):
        """Remove the operator registered under *alias*."""
        delattr(self, alias)

    def decorate(self, alias, *decorators):
        """Wrap the function registered under *alias* with one or more decorators.

        Note: decorating a toolbox function makes it unpicklable, which breaks
        multiprocessing.Pool.map.  Decorate the function before registering it
        (using @notation) if picklability is required.
        """
        pfunc = getattr(self, alias)
        function, args, kargs = pfunc.func, pfunc.args, pfunc.keywords
        for decorator in decorators:
            function = decorator(function)
        self.register(alias, function, *args, **kargs)


class Logbook(list):
    """Evolution records as a chronological list of dictionaries.

    Records are appended via :meth:`record` and queried via :meth:`select`.

    The :attr:`bookshelf` dict holds arbitrary run metadata (counters,
    timing information, etc.) that does not fit the per-generation record
    structure.
    """

    def __init__(self):
        self.buffindex = 0
        self.chapters = defaultdict(Logbook)
        self.bookshelf = {}
        self.columns_len = None
        self.header = None
        self.log_header = True

    def add(self, alias, obj):
        """Store an arbitrary object on the bookshelf under *alias*."""
        self.bookshelf[f'{alias}'] = obj

    def record(self, **infos):
        """Append a dict of key-value pairs as the next log entry.

        Values that are themselves dicts are recorded in named chapters.
        """
        apply_to_all = {k: v for k, v in infos.items() if not isinstance(v, dict)}
        for key, value in list(infos.items()):
            if isinstance(value, dict):
                chapter_infos = value.copy()
                chapter_infos.update(apply_to_all)
                self.chapters[key].record(**chapter_infos)
                del infos[key]
        self.append(infos)

    def select(self, *names):
        """Return per-entry lists for the requested field names.

        With a single name, returns a flat list.
        With multiple names, returns a tuple of lists.
        """
        if len(names) == 1:
            return [entry.get(names[0], None) for entry in self]
        return tuple([entry.get(name, None) for entry in self] for name in names)

    @property
    def stream(self):
        """Return formatted log entries that have not yet been streamed."""
        startindex, self.buffindex = self.buffindex, len(self)
        return self.__str__(startindex)

    def __delitem__(self, key):
        if isinstance(key, slice):
            for i in range(*key.indices(len(self))):
                self.pop(i)
                for chapter in self.chapters.values():
                    chapter.pop(i)
        else:
            self.pop(key)
            for chapter in self.chapters.values():
                chapter.pop(key)

    def pop(self, index=0):
        """Remove and return the entry at *index*, keeping the buffer pointer consistent."""
        if index < self.buffindex:
            self.buffindex -= 1
        return super(self.__class__, self).pop(index)

    def __txt__(self, startindex):
        columns = self.header
        if not columns:
            columns = sorted(self[0].keys()) + sorted(self.chapters.keys())
        if not self.columns_len or len(self.columns_len) != len(columns):
            self.columns_len = [len(c) for c in columns]

        chapters_txt = {}
        offsets = defaultdict(int)
        for name, chapter in self.chapters.items():
            chapters_txt[name] = chapter.__txt__(startindex)
            if startindex == 0:
                offsets[name] = len(chapters_txt[name]) - len(self)

        str_matrix = []
        for i, line in enumerate(self[startindex:]):
            str_line = []
            for j, name in enumerate(columns):
                if name in chapters_txt:
                    column = chapters_txt[name][i + offsets[name]]
                else:
                    value = line.get(name, "")
                    string = "{0:n}" if isinstance(value, float) else "{0}"
                    column = string.format(value)
                self.columns_len[j] = max(self.columns_len[j], len(column))
                str_line.append(column)
            str_matrix.append(str_line)

        if startindex == 0 and self.log_header:
            nlines = 1
            if len(self.chapters) > 0:
                nlines += max(map(len, chapters_txt.values())) - len(self) + 1
            header = [[] for _ in range(nlines)]
            for j, name in enumerate(columns):
                if name in chapters_txt:
                    length = max(len(line.expandtabs()) for line in chapters_txt[name])
                    blanks = nlines - 2 - offsets[name]
                    for i in range(blanks):
                        header[i].append(" " * length)
                    header[blanks].append(name.center(length))
                    header[blanks + 1].append("-" * length)
                    for i in range(offsets[name]):
                        header[blanks + 2 + i].append(chapters_txt[name][i])
                else:
                    length = max(len(line[j].expandtabs()) for line in str_matrix)
                    for line in header[:-1]:
                        line.append(" " * length)
                    header[-1].append(name)
            str_matrix = chain(header, str_matrix)

        template = "\t".join("{%i:<%i}" % (i, k) for i, k in enumerate(self.columns_len))
        return [template.format(*line) for line in str_matrix]

    def __str__(self, startindex=0):
        return "\n".join(self.__txt__(startindex))
