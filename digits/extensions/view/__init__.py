# Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
from __future__ import absolute_import

import copy
from pkg_resources import iter_entry_points

from . import boundingBox
from . import imageGradients
from . import imageOutput
from . import imageSegmentation
from . import rawData

# Entry point group (this is the key we use to register and
# find installed plug-ins)
GROUP = "digits.plugins.view"


# built-in extensions
builtin_view_extensions = [
    boundingBox.Visualization,
    imageGradients.Visualization,
    imageOutput.Visualization,
    imageSegmentation.Visualization,
    rawData.Visualization,
]


def get_default_extension():
    """
    return the default view extension
    """
    return rawData.Visualization


def get_extensions(show_all=False):
    """
    return set of data data extensions
    """
    extensions = copy.copy(builtin_view_extensions)
    # find installed extension plug-ins
    for entry_point in iter_entry_points(group=GROUP, name=None):
        extensions.append(entry_point.load())

    return [extension
            for extension in extensions
            if show_all or extension.get_default_visibility()]


def get_extension(extension_id):
    """
    return extension associated with specified extension_id
    """
    for extension in get_extensions(show_all=True):
        if extension.get_id() == extension_id:
            return extension
    return None
