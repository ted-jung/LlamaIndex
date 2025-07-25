import random

class TourPlanner:

    # Mock data
    # this can be replaced with a vector DB to get relevant data
    # Searching on Vector Database
    def __init__(self):
        self.destinations = {
            "Paris": {
                "activities": ["Eiffel Tower", "Louvre Museum", "Notre Dame Cathedral", "Seine River Cruise", "Montmartre"],
                "food_types": ["French cuisine", "Cafes", "Patisseries"],
                "travel_style_matches": ["romantic", "cultural", "sightseeing"],
                "budget_level": "medium",
                "famous_landmarks": ["Eiffel Tower", "Louvre Museum"]
            },
            "Kyoto": {
                "activities": ["Kinkaku-ji (Golden Pavilion)", "Fushimi Inari-taisha", "Arashiyama Bamboo Grove", "Gion District", "Tea Ceremony"],
                "food_types": ["Japanese cuisine", "Sushi", "Ramen"],
                "travel_style_matches": ["cultural", "historical", "nature"],
                "budget_level": "medium",
                "famous_landmarks": ["Kinkaku-ji", "Fushimi Inari-taisha"]
            },
            "New York City": {
                "activities": ["Statue of Liberty", "Times Square", "Central Park", "Broadway Show", "Museum of Modern Art (MoMA)"],
                "food_types": ["Diverse cuisine", "Street food", "Fine dining"],
                "travel_style_matches": ["urban", "entertainment", "sightseeing"],
                "budget_level": "high",
                "famous_landmarks": ["Statue of Liberty", "Times Square"]
            },
            "Bali": {
                "activities": ["Ubud Monkey Forest", "Tegallalang Rice Terraces", "Seminyak Beach", "Temple Hopping", "Yoga Retreat"],
                "food_types": ["Indonesian cuisine", "Seafood", "Vegan"],
                "travel_style_matches": ["relaxing", "adventure", "nature", "spiritual"],
                "budget_level": "low",
                "famous_landmarks": ["Tanah Lot Temple", "Mount Batur"]
            }
        }
        self.user_preferences = {}


    # =========================================================================
    # Prompts the user for basic trip details. This is where user input is collected.
    # need to make outline based on user input
    # The app will ask you for your destination, number of days, travel style, 
    # budget, and food preferences.
    # One agent for Outline and handoff to other agent for building detail
    # =========================================================================
    def get_user_preferences(self):
        print("Welcome to your AI Tour Planner!")
        print("Let's plan your perfect trip.")

        while True:
            destination = input("Where would you like to go? (e.g., Paris, Kyoto, New York City, Bali): ").strip()
            if destination in self.destinations:
                self.user_preferences["destination"] = destination
                break
            else:
                print("Sorry, I don't have information for that destination yet. Please choose from the available options.")

        self.user_preferences["num_days"] = int(input("How many days will you be traveling? "))
        self.user_preferences["travel_style"] = input("What's your preferred travel style? (e.g., romantic, cultural, adventure, relaxing, urban, historical, nature, sightseeing, entertainment, spiritual): ").strip().lower()
        self.user_preferences["budget"] = input("What's your budget level? (low, medium, high): ").strip().lower()
        self.user_preferences["food_preference"] = input("Any food preferences? (e.g., Japanese cuisine, French cuisine, seafood, vegan): ").strip().lower()


    # Get context from previous agent to generate(write) schedule
    def generate_itinerary(self):
        destination_info = self.destinations.get(self.user_preferences["destination"])
        if not destination_info:
            return "Could not find information for the specified destination."

        itinerary = []
        num_days = self.user_preferences["num_days"]
        selected_activities = []

        # Filter activities based on travel style and unique landmarks
        available_activities = []
        for activity in destination_info["activities"]:
            if self.user_preferences["travel_style"] in destination_info["travel_style_matches"] or \
               activity in destination_info["famous_landmarks"]:
                available_activities.append(activity)

        # Ensure unique activities are chosen for each day
        if len(available_activities) < num_days:
            # If not enough unique activities, just repeat or suggest general exploration
            print(f"Warning: Not enough unique activities for {num_days} days. Some activities might be repeated or general suggestions will be given.")
            selected_activities = available_activities * (num_days // len(available_activities)) + \
                                 available_activities[:(num_days % len(available_activities))]
        else:
            selected_activities = random.sample(available_activities, min(num_days, len(available_activities)))


        for day in range(1, num_days + 1):
            day_plan = f"Day {day}: "
            activity_suggestion = "Explore local sights."
            if selected_activities:
                activity_suggestion = selected_activities.pop(0) # Get a unique activity for the day

            food_suggestion = "Try local cuisine."
            matching_food = [food for food in destination_info["food_types"] if self.user_preferences["food_preference"] in food]
            if matching_food:
                food_suggestion = f"Enjoy {random.choice(matching_food)}."
            elif destination_info["food_types"]:
                food_suggestion = f"Try {random.choice(destination_info['food_types'])}."

            itinerary.append(f"{day_plan} {activity_suggestion}. {food_suggestion}")

        # Add a concluding remark based on budget
        concluding_remark = ""
        if self.user_preferences["budget"] == destination_info["budget_level"]:
            concluding_remark = "This itinerary is tailored to your budget!"
        elif self.user_preferences["budget"] == "low" and destination_info["budget_level"] == "medium":
            concluding_remark = "Consider looking for budget-friendly options for accommodation and dining."
        elif self.user_preferences["budget"] == "high" and destination_info["budget_level"] == "medium":
            concluding_remark = "You have room to splurge on some luxury experiences!"

        return "\n".join(itinerary) + f"\n\nEnjoy your trip to {self.user_preferences['destination']}! {concluding_remark}"


    # Entry to init schedule for traveling
    def run(self):
        self.get_user_preferences()
        itinerary = self.generate_itinerary()
        print("\n--- Your Personalized Itinerary ---")
        print(itinerary)

# Run the app
if __name__ == "__main__":
    app = TourPlanner()
    app.run()