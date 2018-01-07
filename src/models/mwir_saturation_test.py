import numpy as np
import matplotlib.pyplot as plt

def main():

    bg1 = 300  # K
    bg2 = 280  # K

    em1 = 1200  # K
    em2 = 600  # K

    hotspot_sizes = np.arange(1, 10000, 1)  # in sq. m.
    pixel_size = 1e6  # in sq. m.
    frac_areas = hotspot_sizes / pixel_size

    plt.plot(frac_areas, bg1+em1*frac_areas, 'k--')
    plt.plot(frac_areas, np.ones(frac_areas.shape[0])*312)
    #plt.plot(frac_areas, bg2+em1*frac_areas, 'k--')
    #plt.plot(frac_areas, bg1+em2*frac_areas, 'k-')
    #plt.plot(frac_areas, bg2+em2*frac_areas, 'k-')
    plt.show()


if __name__ == "__main__":
    main()
