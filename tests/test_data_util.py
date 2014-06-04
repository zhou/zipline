#
# Copyright 2013 Quantopian, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from collections import deque

import numpy as np

import pandas as pd
import pandas.util.testing as tm

from zipline.utils.data import RollingPanel


class TestRollingPanel(unittest.TestCase):

    def test_basics(self):
        items = ['foo', 'bar', 'baz']
        minor = ['A', 'B', 'C', 'D']

        window = 10

        rp = RollingPanel(window, items, minor, cap_multiple=2)

        dates = pd.date_range('2000-01-01', periods=30, tz='utc')

        major_deque = deque()

        frames = {}

        for i in range(30):
            frame = pd.DataFrame(np.random.randn(3, 4), index=items,
                                 columns=minor)
            date = dates[i]

            rp.add_frame(date, frame)

            frames[date] = frame
            major_deque.append(date)

            if i >= window:
                major_deque.popleft()

            result = rp.get_current()
            expected = pd.Panel(frames, items=list(major_deque),
                                major_axis=items, minor_axis=minor)
            tm.assert_panel_equal(result, expected.swapaxes(0, 1))

    def test_adding_and_dropping_items(self):
        items = list(range(5))
        minor = list(range(10))

        expected_items = list(range(5))
        expected_minor = list(range(10))

        window = 10

        rp = RollingPanel(window, items, minor, cap_multiple=2)

        dates = pd.date_range('2000-01-01', periods=30, tz='utc')

        major_deque = deque()

        frames = {}

        for i in range(30):
            frame = pd.DataFrame(np.random.randn(5, 10), index=items,
                                 columns=minor)

            if i > window:
                # Old labels should start to get dropped
                expected_minor = expected_minor[1:]
                expected_items = expected_items[1:]

            date = dates[i]

            rp.add_frame(date, frame)

            frames[date] = frame
            major_deque.append(date)

            if i >= window:
                major_deque.popleft()

            result = rp.get_current()
            np.testing.assert_array_equal(sorted(result.minor_axis.values),
                                          sorted(expected_minor))
            np.testing.assert_array_equal(sorted(result.items.values),
                                          sorted(expected_items))
            tm.assert_frame_equal(frame.T,
                                  result.ix[frame.index, -1, frame.columns])

            # shift minor and items to trigger updating of underlying data
            # structure
            minor = minor[1:]
            minor.append(minor[-1] + 1)
            items = items[1:]
            items.append(items[-1] + 1)

            expected_minor.append(expected_minor[-1] + 1)
            expected_items.append(expected_items[-1] + 1)


def run_history_implementations(option='clever', n=500, change_fields=False,
                                copy=False):
    items = range(15)
    minor = range(20)
    window = 100
    periods = n

    dates = pd.date_range('2000-01-01', periods=periods, tz='utc')
    frames = {}

    if option == 'clever':
        rp = RollingPanel(window, items, minor, cap_multiple=2)
        major_deque = deque()

        for i in range(periods):
            if change_fields and (i % 5) == 0:
                minor = minor[1:]
                minor.append(minor[-1] + 1)
                items = items[1:]
                items.append(items[-1] + 1)

            dummy = pd.DataFrame(np.random.randn(len(items), len(minor)),
                                 index=items, columns=minor)

            frame = dummy * (1 + 0.001 * i)
            date = dates[i]

            rp.add_frame(date, frame)

            frames[date] = frame
            major_deque.append(date)

            if i >= window:
                del frames[major_deque.popleft()]

            result = rp.get_current()
            if copy:
                result = result.copy()
    else:
        major_deque = deque()
        dummy = pd.DataFrame(np.random.randn(len(items), len(minor)),
                             index=items, columns=minor)

        for i in range(periods):
            frame = dummy * (1 + 0.001 * i)
            date = dates[i]
            frames[date] = frame
            major_deque.append(date)

            if i >= window:
                del frames[major_deque.popleft()]

            result = pd.Panel(frames, items=list(major_deque),
                              major_axis=items, minor_axis=minor)
