import sys  
import os  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json 
from websocietysimulator.llm import LLMBase, InfinigenceLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase 
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
#from websocietysimulator.agent.modules.memory_modules import MemoryBase, MemoryDILU
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import shutil
import uuid
import logging
# import ipdb
import numpy as np
logging.basicConfig(level=logging.INFO)

class MemoryBase:  
    def __init__(self, memory_type: str, llm) -> None:  
        self.llm = llm  
        self.embedding = self.llm.get_embedding_model()  
        db_path = os.path.join('./db', memory_type, f'{str(uuid.uuid4())}')  
        
        if os.path.exists(db_path):  
            shutil.rmtree(db_path)  # Clear previous memory for this instance  
        
        # Use Chroma as the storage backend  
        self.scenario_memory = Chroma(  
            embedding_function=self.embedding,  
            persist_directory=db_path  
        )    

    def __call__(self, current_situation: str = ''):
        if 'review:' in current_situation:
            self.addMemory(current_situation.replace('review:', ''))
        else:
            return self.retriveMemory(current_situation)

    def retriveMemory(self, query_scenario):  
        raise NotImplementedError("This method should be implemented by subclasses.")  

    def addMemory(self, user_data):  
        raise NotImplementedError("This method should be implemented by subclasses.")

class MemoryItem(MemoryBase):  
    def __init__(self, llm):  
        super().__init__(memory_type='item', llm=llm) 
     

    def retriveMemory(self, query_scenario: str):
        # Extract task name from query scenario
        task_name = query_scenario
        
        # Return empty string if memory is empty
        if self.scenario_memory._collection.count() == 0:
            return ''
            
        # Find most similar memory
        similarity_results = self.scenario_memory.similarity_search_with_score(
            task_name, k=1)
            
        # Extract task trajectories from results
        task_trajectories = [
            result[0].metadata['task_trajectory'] for result in similarity_results
        ]
        
        # Join trajectories with newlines and return
        return '\n'.join(task_trajectories)  

    def addMemory(self, current_situation: str):
        # Extract task description
        task_name = current_situation
            
        # Create document with metadata
        memory_doc = Document(
            page_content=task_name,
            metadata={
                "task_name": task_name,
                "task_trajectory": current_situation
            }
        )
            
        # Add to memory store
        self.scenario_memory.add_documents([memory_doc])


class PlanningBaseline(PlanningBase):
    """Inherit from PlanningBase"""
    
    def __init__(self, llm):
        """Initialize the planning module"""
        super().__init__(llm=llm)
    
    def __call__(self, task_description):
        """Override the parent class's __call__ method"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None', 
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['item_id']}
            }
        ]
        return self.plan

class ReasoningBaseline(ReasoningBase):
    """Inherit from ReasoningBase"""
    
    def __init__(self, profile_type_prompt, llm):
        """Initialize the reasoning module"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str, user_style=''):
        """Override the parent class's __call__ method"""
        # print("user_style:", user_style)
        prompt = '''
{task_description}'''
        prompt = prompt.format(task_description=task_description)
        
        messages = [{"role": "user", "content": prompt}]
        reasoning_result = self.llm(
            messages=messages,
            temperature=0.0,
            max_tokens=1000
        )
        return reasoning_result



class MySimulationAgent(SimulationAgent):
    """Participant's implementation of SimulationAgent."""
    
    def __init__(self, llm: InfinigenceLLM):
        """Initialize MySimulationAgent"""
        super().__init__(llm=llm)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm)
        self.memory_item = MemoryItem(llm=self.llm)

    # Process item
    def processItemAmazon(self, item: dict):  
        """Process Amazon item Data"""  
        name = item.get("title", "Unknown Product")  
        features = item.get("features", [])  
        description = item.get("description", ["No description"])[0] if item.get("description") else "No description"  
        rating_number = item.get("rating_number", 0)  
        average_rating = item.get("average_rating", 0)  
        main_category = item.get("main_category", "Uncategorized")  
        
        # 从嵌套的details中提取信息  
        details = item.get("details", {})  
        size = details.get("Size", "Unknown Size")  
        material = details.get("Material", "Unknown Material")  
        brand = details.get("Brand", "Unknown Brand")  
        
        # 购买决策参考字段  
        price = item.get("price") or "Price Not Available"  
        warranty = details.get("Warranty Description", "No warranty information")  

        prompt = f"""  
            Generate short sentences for the product:  

            Product Details:  
            1. Basic Information  
            - Name: {name}  
            - Category: {main_category}  
            - Brand: {brand}

            2. Performance Metrics  
            - Average Rating: {average_rating} / 5  
            - Number of Ratings: {rating_number}  

            3. Key Features:  
            {features}  

            4. Detailed Description:  
            {description}  

            5. Purchase Considerations  
            - Price: {price}  
            - Warranty: {warranty}  

            Analysis Request:  
            Extract and summarize the most important information a potential reviewer would need to know about this product.
            """

        # 调用大语言模型生成分析报告  
        reasoning_result = self.llm(  
            messages=[{"role": "user", "content": prompt}],  
            temperature=0.0,  
            max_tokens=2000  
        )  

        return reasoning_result


    def processItemYelp(self, item: dict):   
        name = item.get("name", "Unknown")  
        stars = item.get("stars", "Unknown")  
        is_open = item.get("is_open", "Unknown")  

        # 确保 attributes 始终是字典，如果是 None 则设为空字典  
        attributes = item.get("attributes", {}) or {}  
        restaurants_reservations = attributes.get("RestaurantsReservations", "Unknown")
        restaurants_good_for_groups = attributes.get("RestaurantsGoodForGroups", "Unknown")
        restaurants_attire = attributes.get("RestaurantsAttire", "casual").replace("'", "")
        business_accepts_credit_cards = attributes.get("BusinessAcceptsCreditCards", "Unknown")
        wi_fi = attributes.get("WiFi", "free").replace("'", "")
        has_tv = attributes.get("HasTV", "Unknown")
        restaurants_take_out = attributes.get("RestaurantsTakeOut", "Unknown")
        ambience = attributes.get("Ambience", "{}")
        good_for_kids = attributes.get("GoodForKids", "Unknown")
        noise_level = attributes.get("NoiseLevel", "Unknown").replace("u'", "").replace("'", "")
        happy_hour = attributes.get("HappyHour", "Unknown")
        restaurants_delivery = attributes.get("RestaurantsDelivery", "Unknown")
        wheelchair_accessible = attributes.get("WheelchairAccessible", "Unknown")
        outdoor_seating = attributes.get("OutdoorSeating", "Unknown")
        restaurants_table_service = attributes.get("RestaurantsTableService", "Unknown")
        hours = item.get("hours", {})  

    # 处理停车信息  
        '''
        try:  
            business_parking_str = safe_get(attributes, "BusinessParking", "{}")  
            business_parking = json.loads(business_parking_str) if business_parking_str else {}  
            
            valet = business_parking.get("valet", False)  
            street_parking = business_parking.get("street", False)  
            lot_parking = business_parking.get("lot", False)  
            garage = business_parking.get("garage", None)  
            validated_parking = business_parking.get("validated", None)  
        except (json.JSONDecodeError, TypeError):  
            valet = False  
            street_parking = False  
            lot_parking = False  
            garage = None  
            validated_parking = None  
        '''
        # Generating the prompt  
        prompt = f"""  
            Please generate a paragraph for this business '{name}'  

            Contextual Data Overview:  
            1. Operational Metrics  
            - Current Status: {'Open' if is_open == 1 else 'Closed'}  
            - Overall Rating: {stars} stars   

            2. Service Capabilities  
            - Reservations: {'Available' if restaurants_reservations == 'True' else 'Not Available'}  
            - Group-Friendly: {'Yes' if restaurants_good_for_groups == 'True' else 'No'}  
            - Table Service: {'Provided' if restaurants_table_service == 'True' else 'Limited'}  
            - Takeout: {'Offered' if restaurants_take_out == 'True' else 'Not Available'}  
            - Delivery: {'Available' if restaurants_delivery == 'True' else 'Not Available'}  

            3. Dining Environment  
            - Ambience Type: {ambience}  
            - Noise Level: {noise_level}  
            - Kid-Friendly: {'Yes' if good_for_kids == 'True' else 'No'}  
            - Dress Code: {restaurants_attire}  
            - Outdoor Seating: {'Available' if outdoor_seating == 'True' else 'Not Available'}  

            4. Facilities and Amenities  
            - WiFi: {wi_fi}  
            - TV: {'Available' if has_tv == 'True' else 'Not Available'}  
            - Wheelchair Access: {'Yes' if wheelchair_accessible == 'True' else 'No'}    

            5. Dining Experience Enhancers    
            - Happy Hour: {'Available' if happy_hour == 'True' else 'Not Offered'}  
            - Payment Methods: {'Credit Cards Accepted' if business_accepts_credit_cards == 'True' else 'Limited Payment Options'}  

            6. Operating Hours  
            {hours}  

            Analytical Task:  
            Generate a comprehensive, narrative-driven report that:  
            - Transforms these raw data points into a compelling profile  
            - Provides nuanced insights beyond basic facts  
            - Captures the unique character of this business  
            """  
        # ipdb.set_trace()
        #print("prompt here:", prompt)

        # Step 3: Call LLM once to generate the full report  
        reasoning_result = self.llm(  
            messages=[{"role": "user", "content": prompt}],  
            temperature=0.0,  
            max_tokens=2000  
        )  

        # Step 4: Return the generated report  
        return reasoning_result

    def processItemGoodreads(self, item: dict):
        """Process Amazon item Data"""
        title = item.get('title', 'Unknown Title')  
        description = item.get('description', 'No description available')  
        format = item.get('format', 'Unknown Format')  
        num_pages = item.get('num_pages', 'Unknown')  
        publisher = item.get('publisher', 'Unknown Publisher')  
        publication_year = item.get('publication_year', 'Unknown Year')
        # 评价指标
        average_rating = item.get('average_rating', 'Unknown')  
        ratings_count = item.get('ratings_count', 'Unknown')  
        text_reviews_count = item.get('text_reviews_count', 'Unkown')

        prompt = f"""  
            Book Information Extraction for Review Preparation  

            1. Book Fundamentals  
            - Title: {title}  
            - Format: {format}  
            - Pages: {num_pages}  
            - Publisher: {publisher}  
            - Publication Year: {publication_year}  

            2. Performance Metrics  
            - Average Rating: {average_rating} / 5  
            - Total Ratings: {ratings_count}  
            - Text Reviews: {text_reviews_count}  

            3. Book Description:  
            {description}  

            Analysis Request:  
            Extract and summarize the most critical information a potential reviewer would need to know about this book, focusing on:  
            - Core themes and content  
            - Potential target audience  
            - Distinctive characteristics  
            - Literary significance  
            - Readability and accessibility  

            Provide a neutral, fact-based summary that captures the essence of the book and gives potential readers a clear understanding of what to expect.  
            """

        # 调用大语言模型生成分析报告  
        reasoning_result = self.llm(  
            messages=[{"role": "user", "content": prompt}],  
            temperature=0.0,  
            max_tokens=2000  
        )  

        return reasoning_result

    def processUsergoodreads(self, reviews_user: list):
        '''
        Get the user style based on the user's historical reviews on Goodreads.
        '''
        # 提取星级评分  
        star_ratings = [review['stars'] for review in reviews_user]  
        
        # 计算平均值和方差  
        mean_rating = np.mean(star_ratings)  
        variance_rating = np.var(star_ratings)  
        
        # 构建初始 prompt，加入平均值和方差信息  
        prompt = (  
            f"Analyze the review style and tendencies of a Goodreads user based on their historical reviews. "  
            f"The user's average star rating is {mean_rating:.2f}, and the variance in their ratings is {variance_rating:.2f}. "  
            "Focus on the following aspects:\n\n"  
            "1. **Star ratings**: Discuss whether the user tends to give higher or lower ratings based on this average. "  
            "Does their variance suggest consistent ratings, or do they tend toward extreme ratings (e.g., only 1 or 5 stars)?\n"  
            "2. **Review sentiment and emotion**: Identify common sentiments (positive, neutral, negative) "  
            "and emotional tones (e.g., enthusiastic, critical, disappointed) in the user's reviews. "  
            "Summarize their review style based on the language used.\n"  
            "3. **Correlation between star ratings and review content**: Analyze what types of books or aspects tend to receive higher ratings, "  
            "and what aspects tend to receive lower ratings. Highlight specific themes or patterns in their reviews that align with their scores.\n\n"  
        )  

        # 遍历用户的每条评论，将内容添加到 prompt  
        for review in reviews_user:  
            prompt += f"Book Review:\n"  
            prompt += f"Stars: {review['stars']} out of 5\n"  
            prompt += f"Review Text: {review['text']}\n\n"  
        
        # 总结部分  
        prompt += (  
            "Based on the provided data, summarize the user's review style, addressing the above dimensions. "  
            "Pay special attention to the relationship between their ratings, the content of their reviews, "  
            "and the emotional tone they exhibit."  
        )     
        reasoning_result = self.llm(  
            messages=[{"role": "user", "content": prompt}],  
            temperature=0.0,  
            max_tokens=2000  
        )  
        
        # Step 4: Return the generated report  
        return reasoning_result  
   
    def processUseramazon(self, reviews_user: list):
        '''
        Get the user style based on the user's historical reviews on Goodreads.
        '''
        # 提取星级评分  
        star_ratings = [review['stars'] for review in reviews_user]  
        
        # 计算平均值和方差  
        mean_rating = np.mean(star_ratings)  
        variance_rating = np.var(star_ratings)  
        
        # 构建初始 prompt，加入平均值和方差信息  
        prompt = (  
            f"You are a user in Amazon, please analyze the review style of yourself based on their historical reviews. "  
            f"Your average star rating is {mean_rating:.2f}, and the variance in the ratings is {variance_rating:.2f}. "  
            "Focus on the following aspects:\n\n"  
            "1. **Star ratings**: Discuss whether you tend to give higher or lower ratings based on this average. "  
            "Does your variance suggest consistent ratings, or do they tend toward extreme ratings (e.g., only 1 or 5 stars)?\n"  
            "2. **Review sentiment and emotion**: Identify common sentiments (positive, neutral, negative) "  
            "and emotional tones (e.g., enthusiastic, critical, disappointed) in your reviews. "  
            "3. **Correlation between star ratings and review content**: Analyze what types of items or aspects tend to receive higher ratings, "  
            "and what aspects tend to receive lower ratings. Highlight specific themes or patterns in your reviews that align with your scores.\n\n"  
        )  

        # 遍历用户的每条评论，将内容添加到 prompt  
        for review in reviews_user:  
            prompt += f"Review:\n"  
            prompt += f"Stars: {review['stars']} out of 5\n"  
            prompt += f"Review Text: {review['text']}\n\n"  
        
        # 总结部分  
        prompt += (  
            "Based on the provided data, summarize your review style, addressing the above dimensions. "  
            "Pay special attention to the relationship between your ratings, the content of your reviews, "  
            "and the emotional tone you exhibit."  
        )  
        reasoning_result = self.llm(  
            messages=[{"role": "user", "content": prompt}],  
            temperature=0.0,  
            max_tokens=2000  
        )  
         
        return reasoning_result
    
    
    # Process user style (only for yelp)
    def processUserYelp(self, user: dict):
        """  
        Retrieve memory for a specific user and analyze their behavior to generate a report.  

        Args:  
            user: User information in dict.  

        Returns:  
            str: A formatted string report summarizing the user's style and tendencies.  
        """   
        
        # Rating tendency analysis  
        average_stars = user.get("average_stars", 0)  
        if average_stars >= 4.0:  
            rating_tendency = f"This user tends to give high ratings (average stars: {average_stars:.1f}), indicating a more generous or satisfied personality."  
        elif average_stars < 3.0:  
            rating_tendency = f"This user tends to give low ratings (average stars: {average_stars:.1f}), suggesting a critical or selective personality."  
        else:  
            rating_tendency = f"This user gives balanced ratings (average stars: {average_stars:.1f}), reflecting a moderate and fair judgment style."  

        # Interaction style analysis  
        funny = user.get("funny", 0)  
        compliment_funny = user.get("compliment_funny", 0)  
        compliment_hot = user.get("compliment_hot", 0)  
        compliment_plain = user.get("compliment_plain", 0)  
        if funny > 10 or compliment_funny > 10:  
            interaction_style = f"This user often displays a humorous tone in their reviews (funny: {funny}, compliment_funny: {compliment_funny})."  
        elif compliment_hot > 10:  
            interaction_style = f"This user writes emotional and engaging reviews (compliment_hot: {compliment_hot})."  
        elif compliment_plain > 10:  
            interaction_style = f"This user prefers a plain and straightforward tone in reviews (compliment_plain: {compliment_plain})."  
        else:  
            interaction_style = "This user's interaction style is neutral or undefined."  

        # Content complexity analysis  
        elite = user.get("elite", False)  
        compliment_writer = user.get("compliment_writer", 0)  
        compliment_note = user.get("compliment_note", 0)  
        if elite or compliment_writer > 10 or compliment_note > 10:  
            content_complexity = f"This user writes with a high degree of complexity and elegance (elite: {elite}, compliment_writer: {compliment_writer}, compliment_note: {compliment_note})."  
        else:  
            content_complexity = "This user keeps reviews simple and easy to read."  

        # Activity level analysis  
        useful = user.get("useful", 0)  
        fans = user.get("fans", 0)  
        if useful > 50 or fans > 10:  
            activity_level = f"This user is highly active and influential (useful: {useful}, fans: {fans})."  
        else:  
            activity_level = "This user has a low to moderate activity level."  

        # Step 2: Combine the analysis into a single prompt for the LLM  
        prompt = (f"Analysis the following user's review style and tendencies based on the provided analysis:\n\n"  
                f"Rating Tendency:\n{rating_tendency}\n\n"  
                f"Interaction Style:\n{interaction_style}\n\n"  
                f"Content Complexity:\n{content_complexity}\n\n"  
                f"Activity Level:\n{activity_level}\n\n"  
                f"Summarize and explain the user's review style in a professional and engaging manner.")  

        # Step 3: Call LLM once to generate the full report  
        reasoning_result = self.llm(  
            messages=[{"role": "user", "content": prompt}],  
            temperature=0.0,  
            max_tokens=2000  
        )  

        # Step 4: Return the generated report  
        return reasoning_result   
    
    '''
    def processUserYelp(self, reviews_user: list):
        
        #user来自user.json
        #reviews_user来自review.json, 是某个user之前的所有reviews
        
        star_ratings = [review['stars'] for review in reviews_user]  
        
        # 计算平均值和方差  
        mean_rating = np.mean(star_ratings)  
        variance_rating = np.var(star_ratings)  

        # 构建初始 prompt，加入平均值和方差信息  
        prompt = (  
            f"Analyze the review style of a Yelp user based on their historical reviews. "  
            f"The user's average star rating is {mean_rating:.2f}, and the variance in their ratings is {variance_rating:.2f}. "  
            "Focus on the following aspects:\n\n"  
            "1. **Star ratings**: Discuss whether the user tends to give higher or lower ratings based on this average. "  
            "Does their variance suggest consistent ratings, or do they tend toward extreme ratings (e.g., only 1 or 5 stars)?\n"  
            "2. **Review sentiment and emotion**: Identify common sentiments (positive, neutral, negative) "  
            "and emotional tones (e.g., enthusiastic, critical, disappointed) in the user's reviews. "  
            "Summarize their review style based on the language used.\n"  
            "3. **Correlation between star ratings and review content**: Analyze what types of business or aspects tend to receive higher ratings, "  
            "and what aspects tend to receive lower ratings. Highlight specific themes or patterns in their reviews that align with their scores.\n\n"  
        )  

        # 遍历用户的每条评论，将内容添加到 prompt  
        for review in reviews_user:  
            prompt += f"Review:\n"  
            prompt += f"Stars: {review['stars']} out of 5\n"  
            prompt += f"Review Text: {review['text']}\n\n"  
        
        # 总结部分  
        prompt += (  
            "Based on the provided data, summarize the user's review style, addressing the above dimensions. "  
            "Pay special attention to the relationship between their ratings, the content of their reviews, "  
            "and the emotional tone they exhibit."  
        )  
        reasoning_result = self.llm(  
            messages=[{"role": "user", "content": prompt}],  
            temperature=0.0,  
            max_tokens=2000  
        )  
         
        return reasoning_result
    '''
    def workflow(self):
        try:
            plan = self.planning(task_description=self.task) # 先规划任务，格式化返回

            for sub_task in plan:
                if 'user' in sub_task['description']:
                    user = self.interaction_tool.get_user(user_id=self.task['user_id']) # 对于sub_task，返回user和item_id
                elif 'business' in sub_task['description']:
                    item = self.interaction_tool.get_item(item_id=self.task['item_id']) # 也可以操作
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id']) # 根据item_id返回对应的reviews（多个）
            if not reviews_item:  # 检查是否有评论  
                print(f"No reviews found for item_id: {self.task['item_id']}")  
                return 

            # 对该item建立review的memory，感觉还有优化空间
            for review in reviews_item:
                review_text = review['text']
                self.memory_item(f'review: {review_text}')
            
            # 从item memory里提取similar reviews
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id']) # 通过user_id来获取这个人之前做的一些review
            
            review_similar = self.memory_item.retriveMemory(f'{reviews_user[0]["text"]}') # 基于这个人的review从memory里取出相关的对item的review

            # 不同task用不同的处理方式和prompt
            if user['source'] == 'yelp':
                user_style = self.processUserYelp(user)
                name = item.get("name", "Unknown")
                stars = item.get("stars", "Unknown")
                is_open = item.get("is_open", "Unknown")

                # Ensure attributes is always a dictionary, default to empty if None
                attributes = item.get("attributes", {}) or {}

                # Extract attributes with fallback values
                restaurants_reservations = attributes.get("RestaurantsReservations", "Unknown")
                RestaurantsPriceRange2 = attributes.get("RestaurantsPriceRange2", "Unknown")

                restaurants_good_for_groups = attributes.get("RestaurantsGoodForGroups", "Unknown")
                restaurants_attire = attributes.get("RestaurantsAttire", "casual").replace("'", "")
                business_accepts_credit_cards = attributes.get("BusinessAcceptsCreditCards", "Unknown")
                wi_fi = attributes.get("WiFi", "free").replace("'", "")
                has_tv = attributes.get("HasTV", "Unknown")
                restaurants_take_out = attributes.get("RestaurantsTakeOut", "Unknown")
                good_for_kids = attributes.get("GoodForKids", "Unknown")
                noise_level = attributes.get("NoiseLevel", "Unknown").replace("u'", "").replace("'", "")
                happy_hour = attributes.get("HappyHour", "Unknown")
                restaurants_delivery = attributes.get("RestaurantsDelivery", "Unknown")
                wheelchair_accessible = attributes.get("WheelchairAccessible", "Unknown")
                outdoor_seating = attributes.get("OutdoorSeating", "Unknown")
                restaurants_table_service = attributes.get("RestaurantsTableService", "Unknown")
                hours = item.get("hours", {})
                
                price_range_map = {
                    "1": "cheap",
                    "2": "affordable",
                    "3": "moderate",
                    "4": "expensive",
                    "5": "luxury"
                }
                price_range_description = price_range_map.get(RestaurantsPriceRange2, "Unknown")

                # Build a more concise prompt
                task_description = f'''
                You are a real human user on Yelp, a platform for crowd-sourced business reviews.
                Here is your Yelp profile and review history style: {user_style}

                You need to write a review for this business. Here is the information:

                Name of this business: '{name}'

                Contextual Data Overview:
                1. Operational Metrics
                - Current Status: {'Open' if is_open == 1 else 'Closed'}
                - Overall Rating: {stars} stars
                '''

                # Add attributes only if they exist
                if attributes:
                    task_description += "\n2. Service Capabilities\n"
                    if restaurants_reservations != "Unknown":
                        task_description += f"- Reservations: {'Available' if restaurants_reservations == 'True' else 'Not Available'}\n"
                    if restaurants_good_for_groups != "Unknown":
                        task_description += f"- Group-Friendly: {'Yes' if restaurants_good_for_groups == 'True' else 'No'}\n"
                    if restaurants_table_service != "Unknown":
                        task_description += f"- Table Service: {'Provided' if restaurants_table_service == 'True' else 'Limited'}\n"
                    if restaurants_take_out != "Unknown":
                        task_description += f"- Takeout: {'Offered' if restaurants_take_out == 'True' else 'Not Available'}\n"
                    if restaurants_delivery != "Unknown":
                        task_description += f"- Delivery: {'Available' if restaurants_delivery == 'True' else 'Not Available'}\n"

                    task_description += "\n3. Environment\n"
                    if noise_level != "Unknown":
                        task_description += f"- Noise Level: {noise_level}\n"
                    if good_for_kids != "Unknown":
                        task_description += f"- Kid-Friendly: {'Yes' if good_for_kids == 'True' else 'No'}\n"
                    if restaurants_attire != "Unknown":
                        task_description += f"- Dress Code: {restaurants_attire}\n"
                    if outdoor_seating != "Unknown":
                        task_description += f"- Outdoor Seating: {'Available' if outdoor_seating == 'True' else 'Not Available'}\n"
                    if price_range_description != "Unknown":
                        task_description += f"- Price: {price_range_description}\n"

                    task_description += "\n4. Facilities and Amenities\n"
                    if wi_fi != "Unknown":
                        task_description += f"- WiFi: {wi_fi}\n"
                    if has_tv != "Unknown":
                        task_description += f"- TV: {'Available' if has_tv == 'True' else 'Not Available'}\n"
                    if wheelchair_accessible != "Unknown":
                        task_description += f"- Wheelchair Access: {'Yes' if wheelchair_accessible == 'True' else 'No'}\n"

                    task_description += "\n5. Dining Experience Enhancers\n"
                    if happy_hour != "Unknown":
                        task_description += f"- Happy Hour: {'Available' if happy_hour == 'True' else 'Not Offered'}\n"
                    if business_accepts_credit_cards != "Unknown":
                        task_description += f"- Payment Methods: {'Credit Cards Accepted' if business_accepts_credit_cards == 'True' else 'Limited Payment Options'}\n"

                # Add operating hours
                if hours:
                    task_description += f"\n6. Operating Hours\n{hours}\n"

                # Add review instructions
                task_description += '''
                Please analyze the following aspects carefully:
                1. **Star Rating**:
                - Be highly critical when assigning star ratings, ensuring they reflect the true quality of the business.
                - Only businesses that significantly exceed expectations in all key areas should receive a 5-star rating.
                - Minor shortcomings should prevent a perfect score.
                - If the business performs adequately but does not stand out, a 3-star rating is appropriate.
                - Reserve 4 stars for businesses that meet high standards but fall short of excellence.
                - If the business fails to meet expectations in key areas or provides subpar service or products, assign a 2-star rating.
                - If the business fails significantly, demonstrates negligence, or provides poor service, assign a 1-star rating.

                2. **Review Text**:
                - The review text should focus on your personal experience and emotional response.
                - Highlight specific details about the business, including positive aspects that make it stand out and any negative aspects that detract from the experience.
                - Aim to write a review that balances honesty with a fair and thoughtful tone.

                3. **User Engagement**:
                Think about how other users might engage with your review in terms of:
                - **Useful**: Is your review informative and helpful to other potential customers?
                - **Funny**: Does your review include any humorous or entertaining elements?
                - **Cool**: Is your review particularly insightful or praiseworthy?

                **Requirements**:
                - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
                - Use the following guidelines to determine the rating:
                - Businesses delivering exceptional service and products should receive high scores.
                - Be critical when businesses fail to meet basic standards, ensuring ratings reflect the severity of shortcomings.
                - Review text should be 2-4 sentences, focusing on specific details about the experience. While the star rating reflects a critical evaluation, the review can highlight positive elements or provide constructive feedback.

                **Format your response exactly as follows**:
                stars: [your rating]
                review: [your review]
                '''

            elif user['source'] == 'amazon':
                name = item.get("title", "Unknown Product")  
                features = item.get("features", [])  
                description = item.get("description", ["No description"])[0] if item.get("description") else "No description"  
                #rating_number = item.get("rating_number", 0)  
                average_rating = item.get("average_rating", 0)  
                main_category = item.get("main_category", "Uncategorized")  
                
                # 从嵌套的details中提取信息  
                details = item.get("details", {})  
                #size = details.get("Size", "Unknown Size")  
                #material = details.get("Material", "Unknown Material")  
                brand = details.get("Brand", "Unknown Brand")  
                
                # 购买决策参考字段  
                price = item.get("price") or "Price Not Available"  
                #warranty = details.get("Warranty Description", "No warranty information")  

                user_style = self.processUseramazon(reviews_user)
                task_description = f'''
                You are an Amazon customer leaving a review based on your personal experience. Your review should reflect the style and tone of a genuine Amazon user, consistent with your review history. 
                Here is your review style: {user_style}

                Here's the reference product information:

                ### Product Details:  
                1. **Basic Information**:  
                - Name: {name}  
                - Category: {main_category}  
                - Brand: {brand}  

                2. **Performance Metrics**:  
                - Average Rating: {average_rating} / 5  

                3. **Key Features**:  
                {features}  

                4. **Detailed Description**:  
                {description}  

                5. **Purchase Considerations**:  
                - Price: {price}  

                ### Additional Context:  
                - Other users have reviewed this product before: {review_similar}  

                ---
                Please analyze the following aspects carefully:
                1. Based on the item information and your review profile, what rating would you give this product? Remember that many users give 5-star ratings for excellent books that exceed expectations.
                2. Given the product details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this book stand out or negative aspects that severely impact the experience.
                3. Consider how other users might engage with your review in terms of:
                - Useful: How informative and helpful is your review?
                - Funny: Does your review have any humorous or entertaining elements?
                - Cool: Is your review particularly insightful or praiseworthy?

                Requirements:
                1. **Star Rating**:
                - Be highly critical when assigning star ratings, ensuring they reflect the true quality of the product.
                - Only products that significantly exceed expectations in all key areas should receive a 5-star rating.
                - Minor shortcomings should prevent a perfect score.
                - If the product performs adequately but does not stand out, a 3-star rating is appropriate.
                - Reserve 4 stars for products that meet high standards but fall short of excellence.
                - If the product fails to meet expectations in key areas or provides subpar service or products, assign a 2-star rating.
                - If the product fails significantly, demonstrates negligence, or provides poor service, assign a 1-star rating.
                - Review text should be 2-4 sentences, focusing on your personal experience and emotional response.

                - Maintain consistency with your historical review style and rating patterns
                - Focus on specific details about the product rather than generic comments
                - Be generous with ratings when product deliver quality service and products
                - Be critical when product fail to meet basic standards

                Format your response exactly as follows:
                stars: [your rating]
                review: [your review] 
                '''

            elif user['source'] == 'goodreads':
                # item_info = self.processItemGoodreads(item)
                user_style = self.processUsergoodreads(reviews_user)
                title = item.get('title', 'Unknown Title')  
                description = item.get('description', 'No description available')  
                #format = item.get('format', 'Unknown Format')  
                #num_pages = item.get('num_pages', 'Unknown')  
                publisher = item.get('publisher', 'Unknown Publisher')  
                publication_year = item.get('publication_year', 'Unknown Year')
                
                average_rating = item.get('average_rating', 'Unknown')  
                #ratings_count = item.get('ratings_count', 'Unknown')  
                #text_reviews_count = item.get('text_reviews_count', 'Unkown')
                task_description = f'''
                You are a real human user on goodreads, a platform for crowd-sourced book reviews. Here is your review style and history: {user_style}

                You need to write a review for this book, here is the basic information that you can refer to: 

                1. Book Fundamentals  
                - Title: {title}  
                - Publisher: {publisher}  
                - Publication Year: {publication_year}  

                2. Performance Metrics  
                - Average Rating: {average_rating} / 5  

                3. Book Description:  
                {description}  

                ---

                Others have reviewed this book before: {review_similar}

                Please analyze the following aspects carefully:
                1. Based on the item information and your review profile, what rating would you give this book? Remember that many users give 5-star ratings for excellent books that exceed expectations.
                2. Given the book details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this book stand out or negative aspects that severely impact the experience.
                3. Consider how other users might engage with your review in terms of:
                - Useful: How informative and helpful is your review?
                - Funny: Does your review have any humorous or entertaining elements?
                - Cool: Is your review particularly insightful or praiseworthy?

                Requirements:
                1. **Star Rating**:
                - Be highly critical when assigning star ratings, ensuring they reflect the true quality of the book.
                - Only books that significantly exceed expectations in all key areas should receive a 5-star rating.
                - Minor shortcomings should prevent a perfect score.
                - If the book performs adequately but does not stand out, a 3-star rating is appropriate.
                - Reserve 4 stars for books that meet high standards but fall short of excellence.
                - If the book fails to meet expectations in key areas or provides subpar service or products, assign a 2-star rating.
                - If the book fails significantly, demonstrates negligence, or provides poor service, assign a 1-star rating.
                - Review text should be 2-4 sentences, focusing on your personal experience and emotional response.

                - Maintain consistency with your historical review style and rating patterns
                - Focus on specific details about the book rather than generic comments
                - Be generous with ratings when book deliver quality service and products
                - Be critical when book fail to meet basic standards

                Format your response exactly as follows:
                stars: [your rating]
                review: [your review]
                '''



            result = self.reasoning(task_description)
            
            try:
                stars_line = [line for line in result.split('\n') if 'stars:' in line][0]
                review_line = [line for line in result.split('\n') if 'review:' in line][0]
            except:
                print('Error:', result)

            stars = float(stars_line.split(':')[1].strip())
            review_text = review_line.split(':')[1].strip()

            if len(review_text) > 512:
                review_text = review_text[:512]
                
            return {
                "stars": stars,
                "review": review_text
            }
        except Exception as e:
            print(f"Error in workflow: {e}")
            return {
                "stars": 0,
                "review": ""
            }
