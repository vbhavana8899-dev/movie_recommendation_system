import os
import sys
import textwrap

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.recommender import MovieRecommender

# ── ANSI colour helpers ────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"

def c(text, *codes): return "".join(codes) + str(text) + RESET
def header(text):    print(f"\n{c(text, BOLD, CYAN)}")
def success(text):   print(f"{c(text, GREEN)}")
def warn(text):      print(f"{c(text, YELLOW)}")
def error(text):     print(f"{c(text, RED)}")
def divider():       print(c("─" * 55, DIM))

# ── Banner ─────────────────────────────────────────────────────────────────────
BANNER = f"""
{c('╔══════════════════════════════════════════════╗', CYAN, BOLD)}
{c('║   🎬  Movie Recommendation System            ║', CYAN, BOLD)}
{c('║       Content-Based Filtering Engine         ║', CYAN, BOLD)}
{c('╚══════════════════════════════════════════════╝', CYAN, BOLD)}
"""

MENU = f"""
{c('  [1]', YELLOW)} Get movie recommendations
{c('  [2]', YELLOW)} Search for a movie
{c('  [3]', YELLOW)} View movie details
{c('  [4]', YELLOW)} List all movies
{c('  [5]', YELLOW)} About / How it works
{c('  [0]', RED  )} Exit
"""

# ── Helpers ────────────────────────────────────────────────────────────────────
def paginate(df, page_size=10):
    """Print a DataFrame in pages."""
    total = len(df)
    start = 0
    while start < total:
        end = min(start + page_size, total)
        print(df.iloc[start:end].to_string(index=False))
        start += page_size
        if start < total:
            inp = input(f"\n{c('[Enter] next page  [q] quit', DIM)}: ").strip().lower()
            if inp == "q":
                break
        print()

def print_recommendations(recs, source_title):
    header(f"  Recommendations for: {source_title}")
    divider()
    for _, row in recs.iterrows():
        sim_bar = "█" * int(row["similarity"] * 20)
        print(
            f"  {c(row['rank'], BOLD, YELLOW)}. {c(row['title'], BOLD)}\n"
            f"     Genres   : {c(row['genres'], CYAN)}\n"
            f"     Director : {row['director']}\n"
            f"     Match    : {c(sim_bar, GREEN)} {c(row['similarity'], DIM)}\n"
        )

def print_movie_details(movie):
    header(f"  {movie['title']}")
    divider()
    print(f"  {c('Genres   :', BOLD)}  {c(movie['genres'], CYAN)}")
    print(f"  {c('Director :', BOLD)}  {movie['director']}")
    print(f"  {c('Cast     :', BOLD)}  {movie['cast']}")
    wrapped = textwrap.fill(movie['overview'], width=55, initial_indent="  ", subsequent_indent="  ")
    print(f"  {c('Overview :', BOLD)}\n{wrapped}")

def about():
    header("  How it works")
    divider()
    print(textwrap.dedent("""
      This system uses Content-Based Filtering:

      1. FEATURE EXTRACTION
         Each movie's genres, director, cast, and
         overview are combined into a text "soup".

      2. TF-IDF VECTORISATION
         The text soup is converted to a numerical
         vector using TF-IDF, which weighs words by
         how unique they are across all movies.

      3. COSINE SIMILARITY
         The angle between two movie vectors gives
         a similarity score (0 = unrelated, 1 = identical).

      4. RANKING
         Movies closest in vector space to your
         chosen movie are returned as recommendations.
    """))

# ── Main loop ──────────────────────────────────────────────────────────────────
def main():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "movies.csv"
    )

    print(BANNER)
    print(c("  Loading dataset and building similarity matrix...", DIM))
    recommender = MovieRecommender(data_path)
    success(f"  ✓ Loaded {len(recommender.df)} movies.\n")

    while True:
        print(MENU)
        choice = input(c("  Enter choice: ", BOLD)).strip()

        # ── 1. Recommendations ─────────────────────────────────────────────
        if choice == "1":
            query = input(c("\n  Enter movie title (or part of it): ", BOLD)).strip()
            if not query:
                warn("  No input provided."); continue

            try:
                n_input = input(c("  How many recommendations? [default 5]: ", BOLD)).strip()
                top_n = int(n_input) if n_input.isdigit() else 5
            except ValueError:
                top_n = 5

            result = recommender.get_recommendations(query, top_n=top_n)
            if isinstance(result, pd.DataFrame) if not isinstance(result, tuple) else False:
                error(f"  No movie found matching '{query}'.")
            else:
                recs, source = result
                if recs.empty:
                    error(f"  No movie found matching '{query}'.")
                else:
                    print_recommendations(recs, source)

        # ── 2. Search ──────────────────────────────────────────────────────
        elif choice == "2":
            query = input(c("\n  Search title: ", BOLD)).strip()
            matches = recommender.search_movie(query)
            if matches.empty:
                error(f"  No results for '{query}'.")
            else:
                header(f"  Search results for '{query}'")
                divider()
                paginate(matches)

        # ── 3. Movie details ───────────────────────────────────────────────
        elif choice == "3":
            query = input(c("\n  Enter movie title: ", BOLD)).strip()
            movie = recommender.get_movie_details(query)
            if movie is None:
                error(f"  No movie found matching '{query}'.")
            else:
                print_movie_details(movie)

        # ── 4. List all ────────────────────────────────────────────────────
        elif choice == "4":
            header("  All Movies in Dataset")
            divider()
            paginate(recommender.list_all_movies())

        # ── 5. About ───────────────────────────────────────────────────────
        elif choice == "5":
            about()

        # ── 0. Exit ────────────────────────────────────────────────────────
        elif choice == "0":
            print(c("\n  Goodbye! 🎬\n", CYAN, BOLD))
            sys.exit(0)

        else:
            warn("  Invalid choice. Please enter 0–5.")


if __name__ == "__main__":
    import pandas as pd
    main()
