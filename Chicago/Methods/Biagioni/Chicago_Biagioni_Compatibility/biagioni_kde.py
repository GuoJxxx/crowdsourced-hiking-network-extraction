from __future__ import annotations

import getopt
import sys
from itertools import tee
from pathlib import Path

import cv2
import numpy as np

from location import TripLoader
from pylibs import spatialfunclib


cell_size = 1
gaussian_blur = 17
trips_path = "trips/trips_1m/"


def pairwise(iterable):
    first, second = tee(iterable)
    next(second, None)
    return zip(first, second)


class KDE:
    def create_kde_with_trips(self, all_trips):
        if not all_trips:
            raise ValueError("No trips were supplied")
        print(f"trips path: {trips_path}")
        print(f"cell size: {cell_size}")
        print(f"gaussian blur: {gaussian_blur}")
        print("\nFinding bounding box... ", end="", flush=True)

        latitudes = [point.latitude for trip in all_trips for point in trip.locations]
        longitudes = [point.longitude for trip in all_trips for point in trip.locations]
        min_lat, max_lat = min(latitudes) - 0.003, max(latitudes) + 0.003
        min_lon, max_lon = min(longitudes) - 0.005, max(longitudes) + 0.005
        print("done.")

        diff_lat = max_lat - min_lat
        diff_lon = max_lon - min_lon
        width = max(
            1,
            int(diff_lon * spatialfunclib.METERS_PER_DEGREE_LONGITUDE / cell_size),
        )
        height = max(
            1,
            int(diff_lat * spatialfunclib.METERS_PER_DEGREE_LATITUDE / cell_size),
        )
        yscale = height / diff_lat
        xscale = width / diff_lon
        print(f"raster size: {width} x {height}")

        density = np.zeros((height, width), dtype=np.uint16)
        drawing = np.zeros((height, width), dtype=np.uint8)
        for trip_index, trip in enumerate(all_trips, start=1):
            if trip_index % 10 == 0 or trip_index == len(all_trips):
                print(
                    f"\rCreating histogram (trip {trip_index}/{len(all_trips)})... ",
                    end="",
                    flush=True,
                )
            temporary = np.zeros((height, width), dtype=np.uint8)
            for origin, destination in pairwise(trip.locations):
                oy = height - int(yscale * (origin.latitude - min_lat))
                ox = int(xscale * (origin.longitude - min_lon))
                dy = height - int(yscale * (destination.latitude - min_lat))
                dx = int(xscale * (destination.longitude - min_lon))
                cv2.line(temporary, (ox, oy), (dx, dy), 32, 1, cv2.LINE_AA)
                cv2.line(drawing, (ox, oy), (dx, dy), 255, 1, cv2.LINE_AA)
            np.add(density, temporary, out=density, casting="unsafe")
        print("done.")

        cv2.imwrite("raw_data.png", drawing)
        print("Smoothing... ", end="", flush=True)
        smoothed = cv2.GaussianBlur(
            density,
            (gaussian_blur, gaussian_blur),
            sigmaX=0,
            borderType=cv2.BORDER_REPLICATE,
        )
        cv2.imwrite("kde.png", smoothed)
        Path("bounding_box.txt").write_text(
            f"{min_lat} {min_lon} {max_lat} {max_lon}\n",
            encoding="utf-8",
        )
        print("done.\n\nKDE generation complete.")


if __name__ == "__main__":
    options, _ = getopt.getopt(sys.argv[1:], "c:b:p:h")
    for option, value in options:
        if option == "-c":
            cell_size = int(value)
        elif option == "-b":
            gaussian_blur = int(value)
        elif option == "-p":
            trips_path = str(value)
        elif option == "-h":
            print("Usage: kde.py [-c cell_size] [-b blur_size] [-p trips_path]")
            raise SystemExit(0)
    KDE().create_kde_with_trips(TripLoader.load_all_trips(trips_path))
