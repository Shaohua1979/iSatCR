import os
import numpy as np

class TLEGenerator:
    def __init__(self, output_file, planes, satellites_per_plane, inclo, altitude,argpo_init=90,nodeo_phase=None,argpo_phase=True):
        self.output_file = output_file
        self.satellites_per_plane = satellites_per_plane
        self.planes = planes
        self.inclo = inclo
        self.altitude = altitude
        self.mean_motion = self.mean_motion(altitude)
        self.ecco = "0002000"
        self.bstar = "12222-5"
        self.argpo_init=argpo_init
        self.nodeo_phase=nodeo_phase
        self.argpo_phase=argpo_phase

    def mean_motion(self,altitude):
        earth_radius = 6371.0
        gravitational_constant = 398600.4418
        semi_major_axis = earth_radius + altitude
        period = 2 * np.pi * np.sqrt((semi_major_axis ** 3) / gravitational_constant)
        period_days = period / (60 * 60 * 24)
        mean_motion = 1 / period_days
        return mean_motion

    def tle_checksum(self, line):
        checksum = 0
        for c in line[:-1]:
            if c.isdigit():
                checksum += int(c)
            if c == '-':
                checksum += 1
        return checksum % 10

    def generate_tles(self):
        with open(self.output_file, "w") as f:
            for i in range(self.planes):
                nodeo = (180 if self.inclo>=75 else 360) / self.planes * i + (0 if not self.nodeo_phase else 360/ self.planes/self.nodeo_phase[1]*self.nodeo_phase[0])
                for j in range(self.satellites_per_plane):
                    sat_name = f"Satellite1_{self.altitude}_{i+1}_{j+1}"
                    f.write(sat_name + "\n")
                    line1 = "1 {0:05d}U 00000A   23121.00000000  .00000000  00000+0  {1} 0  999".format(i * 100 + j + 1, self.bstar)
                    line1 += str(self.tle_checksum(line1 + " ")) + "\n"
                    f.write(line1)
                    mo = 360.0 / self.satellites_per_plane * j
                    argpo = self.argpo_init  if i%2==0 or not self.argpo_phase else self.argpo_init-180/self.satellites_per_plane
                    line2 = "2 {0:05d} {1:8.4f} {2:8.4f} {3} {4:8.4f} {5:8.4f} {6:11.8f}  999".format(i * 100 + j + 1, self.inclo, nodeo, self.ecco, argpo, mo, self.mean_motion)
                    line2 += str(self.tle_checksum(line2 + " ")) + "\n"
                    f.write(line2)
        print(f"TLEs written to {os.path.abspath(self.output_file)}")

tle_gen = TLEGenerator("Satellite_Data/60Degree_500_12x24_tles_1.txt", 12, 24, 60,500,argpo_init=90,nodeo_phase=None,argpo_phase=True)
tle_gen.generate_tles()