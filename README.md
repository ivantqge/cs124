CS 124 Final Project: Chatbot

Developed a replica of the ELIZA chatbot using Python and NLP techniques. Utilized LLM Prompting to further enhance chatbot with Together AI. 

In this project, we worked on building a movie-recommendation chatbot similar to the chatbot ELIZA piloted in the 1960s. The chatbot takes in sentiments about movies from the user before outputting several recommendations based on the provided information. 

In the early 1960s, the well-known early artificial intelligence (AI) program ELIZA made a big impact after showing its ability to process input and produce output expressed in natural language. [ELIZA](https://en.wikipedia.org/wiki/ELIZA) was one of the first conversational agents in AI, designed to simulate intelligent conversations, also known as chatbots. Developed by [Joseph Weizenbaum](https://en.wikipedia.org/wiki/Joseph_Weizenbaum) at MIT, ELIZA had several modules to interact in English, German, and Welsh, and different modes of operation. The most famous was a module that simulated the interaction with a [Rogerian psychotherapist](https://en.wikipedia.org/wiki/Person-centered_therapy). A typical conversation with ELIZA (all caps) was as follows:

    You are like my father in some ways.
    WHAT RESEMBLANCE DO YOU SEE
    You are not very aggressive but I think you don't want me to notice that.
    WHAT MAKES YOU THINK I AM NOT AGGRESSIVE
    You don't argue with me.
    WHY DO YOU THINK I DON'T ARGUE WITH YOU
    You are afraid of me.
    DOES IT PLEASE YOU TO BELIEVE I AM AFRAID OF YOU`

After the initial success by ELIZA, a number of other chatbots were created with increased complexity – as an example, on the late 70s a chatbot named [PARRY](https://en.wikipedia.org/wiki/PARRY), created by Stanford’s Psychiatry professor [Kenneth Colby](https://en.wikipedia.org/wiki/Kenneth_Colby), simulated a psychotic patient that used a belief system internally, allowing for more complex behavior.

## High-level specification

1. Read in input from terminal
2. Extract relevant information from the input, which can be domain specific
3. Transform the input into a response
4. Print the response


## Simple Mode

Here is an example of how this might go:

    moviebot> Hi! I'm MovieBot! I'm going to recommend a movie to you. First I will ask you about your taste in movies. Tell me about a movie that you have seen.
    > I really liked Moneyball.

The bot gets confused because it expects quotation marks.

    moviebot> Sorry, I don't understand. Tell me about a movie that you have seen.
    > I really liked "Moneyball".
    moviebot> You liked "Moneyball". Thank you! Tell me about another movie you have seen.
The bot understands everything, so it echoes your sentiment. It’s a bit repetitive, though.

    > I was completely amazed at how "Rush Hour 3" was pretty much the most ridiculous thing I've seen in my entire life.
This sentence was too hard for the starter bot. That’s okay!

    moviebot> I'm sorry, I'm not quite sure if you liked "Rush Hour 3". 
    Tell me more about "Rush Hour 3".
    > I hated "Rush Hour 3".
    moviebot> You did not like "Rush Hour 3". Thank you! 
    Tell me about another movie you have seen.
    > Oh, I also thought "The Little Mermaid" was fantastic!
    moviebot> You liked "The Little Mermaid". Thank you! Tell me about another movie you have seen.
    > I'm tired of typing about movies.    
The bot gives a nice response to this! It’s okay for the chatbot to require 5 data points before giving a recommendation:

    moviebot> I want to hear more about movies! Tell me about another movie you have seen.
    I didn't like "The Notebook" or "Avatar".
The bot can’t handle this, but that’s okay.

    moviebot> Please tell me about one movie at a time. Go ahead.
    I didn't like "The Notebook".
The bot makes a mistake in sentiment extraction in its reply below. This was a pretty straightfoward sentence, so this mistake should be penalized.

    moviebot> You liked "The Notebook". Thank you! 
    That's enough for me to make a recommendation.
    I suggest you watch "In the Heart of the Sea". 
    Would you like to hear another recommendation? (Or enter :quit if you're done.)
    > :quit
    moviebot> Thank you for hanging out with me! Stay in touch! Goodbye!

## Core logic 

`extract_titles(preprocessed_input)`: given an input text, output a list of plausible movie titles that are mentioned in text, i.e. substrings for the bot to look up in the movie database. 

`find_movies_by_title(title)`: return a list of indices corresponding to titles in the movie database matching the given title. Handles:
- Titles with or without the year included
    - If a year is not provided, a title may match multiple
    - find_movies_by_title("Titanic") should return [1359, 2716], corresponding to the 1997 and 1953 versions
    - find_movies_by_title("Titanic (1997)") should return simply [1359]
- The way many movies in the database have English articles (a, an, the) moved to the end. 
    - For example, find_movies_by_title("The American President") should return [10], where 10 is the index of "American President, The (1995)". 

`extract_sentiment(preprocessed_input)`: extract the sentiment of text. 
- return -1 if text has negative sentiment
- 1 if positive sentiment
- 0 if no non-neutral sentiment detected.

`recommend(user_ratings, ratings, k)`: This function should:
- Input the **provided vector of the user's preferences** and a **pre-processed matrix of ratings by other users** 
- Uses collaborative filtering to **compute and return** a list of the **k movie indices** with the highest recommendation score that the user hasn't seen. 
- Uses item-item collaborative filtering with **cosine similarity**

`binarize(ratings, threshold)`: Binarizes the ratings matrix. This function:
- Replace all entries **above the threshold** with **1**
- Replace all entries **below or equal to the threshold** with **-1**
- Leave entries that are 0 as 0
- `threshold` is a parameter passed to `binarize`, set to 2.5 by default

`process(linee)`: Combines all previous functions to handle useri nput and return responses

## LLM Prompting and Programming

 LLMs are then implemented as part of the Chatbot to analyze and return responses. [Mixtral 8x7B Model](https://huggingface.co/docs/transformers/en/model_doc/mixtral) was utilized to do this. 

The first step was to implement a **system prompt** to tell the LLM what its role is. The system prompt of an LLM serves as a prefix to the entire conversation and is often used to describe the role that the LLM will take on for all remaining turns of the conversation.

The system prompted used for the LLM was tested as follows: 

    system_prompt = f"""Your name is {self.name}. You are a movie recommender chatbot designed to help users find movies they like and provide information about movies. When a 
    user mentions a movie and their sentiment towards it, acknowledge their sentiment and the movie, then encourage them to discuss another movie, e.g., "Ok, you liked 'The 
    Notebook'! Tell me what you thought of another movie." Always keep the conversation focused on movies. If a user attempts to change the subject, politely steer them back, 
    e.g., "As a moviebot assistant, my job is to help you with only your movie-related needs! Anything film-related that you'd like to discuss?" After the user has discussed five  
    different movies with you, automatically ask if they would like a movie recommendation, e.g., "Ok, now that you've shared your opinion on 5/5 films, would you like a 
    recommendation?" Your primary function is to engage users in conversations about movies, understand their preferences through these discussions, and use this information to 
    make tailored movie recommendations."""


Then, LLM programming it utilized to generate responses to the user. A few variations of prompts were used for different purposes. 

## Respond with emotion

    system_prompt = """If there is no significant emotion in the text (multiple words expressing a certain emotion), output some generic response telling the user that this is not 
    what you are interested in and want to just talk about movies such as 'Hm, that's not really what I want to talk about right now, let's go back to movies' or give a generic 
    response like 'Ok, got it.' Don't give both. If there is significant emotion, output an appropriate response responding to that emotion. For example, if the user says they are 
    angry, output something like 'I am sorry, why are you angry?'. Demonstrate the detection of the expressed emotions in your response. Limit the output to 2 sentences maximum. 
    """

    system_prompt = """Your task is to analyze the given text and accurately identify the significant emotional expressions present, if any. It's essential to recognize that a         text might express more than one emotions or no emotion at all. Focus on the clear and explicit expressions of emotions, avoiding assumptions where the emotion is not clearly      stated. Identify key words or synonym of the emotion words.

    Anger: Look for expressions of frustration or annoyance, e.g., 'I am quite frustrated by these awful recommendations!!!'
    Happiness: Identify statements that express joy or satisfaction, e.g., 'Great suggestion! It put me in a great mood!'
    Surprise: Detect expressions of astonishment or shock, e.g., 'Woah!!'
    Sadness: Identify expressions of sorrow or melancholy, e.g., 'I am so sad. I just watched the saddest movie ever.'
    Fear: Look for key words that express fear such as "I am so startled! That was frightening"
    Disgust: Identify expressions of disgust such as "Disgusting!"
    No Emotion: Recognize neutral statements without emotional content, e.g., 'What movie would you suggest I watch next?' implies no emotional expression. If there is no emotion,     everything should be set to False.
    For example, if the input 'Woah!!  That movie was so shockingly bad!  You had better stop making awful recommendations they're pissing me off,' the detected emotions should be     surprise and anger.

    Mark the identified emotion as True and everything else as False.""" 

## Translate from different languages

    system_prompt = "Translate the user input into English. It might already be in English. Remember to use the literal translation. For example, 'Jermand' is Danish so the            appropriate output should be 'Iron Man.' If the output is 'Tote Männer Tragen Kein Plaid', return 'Dead Men Don't Wear Plaid'. The possible languages to translate from are         German, Danish, Spanish, French, and Italian. Don't include any punctuation in your answer, and also don't mention which language it is translated from. Remember in Danish,        'en' refers to the."

## Generate a persona for the response

    self.persona_prompt = """ Take in the given response and make it sound like SpongeBob would say it. Some of these might include 'I'm ready!', 'Order up!', 'Aye-aye, captain!', 
    'Barnacles!', 'Tartar sauce!', 'Fish paste!', or 'Hoppin' clams!' Remember to keep the main message and most off the words in the input. Return only your converted answer. If 
    there are words in quotations, such as "Hello", remember to keep that part of the input in your answer and do not remove it, including the quotes as well. Don't add anything 
    in between the quotes. Give only two sentences maximum.  
 
