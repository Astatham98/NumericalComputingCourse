import cProfile, pstats
from main import w1_main, w2_main, w3_main

def profiling(profile_string='w1_main(1024)', output_string='w1'):
    """Profiles a given function and outputs .prof files for all and cumultive stats

    Args:
        profile_string (str, optional): String for the function name and arguments. Defaults to 'w1_main(1024)'.
        output_string (str, optional): String to differentiate output file. Defaults to 'w1'.
    """
    cProfile.run(profile_string, 
                output_string + '.prof')


    stats = pstats.Stats(output_string + '.prof')
    stats.sort_stats ('cumulative')
    stats.print_stats(10)
    stats.dump_stats('stats_' + output_string + '.prof')

if __name__ == "__main__":
    profiling('w1_main(1024)', 'w3')
    profiling('w2_main(1024)', 'w2')