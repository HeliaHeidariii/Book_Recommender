import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import messagebox, ttk
from ttkthemes import ThemedTk

# Data
data = {
    'Book': ['To Kill a Mockingbird', '1984', 'Pride and Prejudice', 'The Great Gatsby', 'Harry Potter and the Sorcerer\'s Stone',
             'The Catcher in the Rye', 'The Lord of the Rings', 'Animal Farm', 'To the Lighthouse', 'The Hobbit',
             'The Grapes of Wrath', 'Brave New World', 'Jane Eyre', 'The Odyssey', 'Wuthering Heights', 'The Picture of Dorian Gray',
             'Crime and Punishment', 'Moby-Dick', 'The Adventures of Huckleberry Finn', 'The Brothers Karamazov'],
    'Genre': ['Fiction', 'Dystopian Fiction', 'Romance', 'Classic', 'Fantasy',
              'Coming-of-Age', 'Fantasy', 'Satire', 'Modernist Literature', 'Fantasy',
              'Literary Fiction', 'Dystopian Fiction', 'Gothic Fiction', 'Epic Poetry', 'Gothic Fiction', 'Gothic Fiction',
              'Psychological Fiction', 'Adventure', 'Adventure', 'Philosophical Fiction']
}

# Creating DataFrame
df = pd.DataFrame(data)

# Calculate similarity matrix
count = CountVectorizer()
count_matrix = count.fit_transform(df['Book'])
cosine_sim = cosine_similarity(count_matrix, count_matrix)


def recommend(book, cosine_sim=cosine_sim):
    recommended_books = []
    idx = df[df['Book'] == book].index[0]
    score_series = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top_2_indexes = list(score_series.iloc[1:3].index)

    for i in top_2_indexes:
        recommended_books.append(df.iloc[i]['Book'])

    return recommended_books


def show_genre_recommendations(genre):
    genre_books = df[df['Genre'] == genre]['Book'].tolist()
    if genre_books:
        messagebox.showinfo("Recommended Books", f"Recommended books in {
                            genre} genre: {', '.join(genre_books)}")
    else:
        messagebox.showwarning("Warning", f"No books found in {genre} genre.")


def show_recommendations():
    selected_genre = genre_var.get()
    if selected_genre:
        show_genre_recommendations(selected_genre)
    else:
        messagebox.showwarning("Warning", "Please select a genre.")


# Creating themed Tkinter window
root = ThemedTk(theme="radiance")
root.title("Book Recommender")
root.configure(bg='gray')

# Dropdown menu for genres
genres = df['Genre'].unique().tolist()
genre_var = tk.StringVar(root)
genre_var.set(genres[0])
genre_menu = ttk.OptionMenu(root, genre_var, *genres)
genre_menu.pack()

# Button to show recommendations
recommend_button = ttk.Button(
    root, text="Show Recommendations", command=show_recommendations)
recommend_button.pack()

# Set window size
root.geometry("300x100")

# Run the application
root.mainloop()
