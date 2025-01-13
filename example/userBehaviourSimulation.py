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
logging.basicConfig(level=logging.INFO)

class MemoryBase:  
    def __init__(self, memory_type: str, llm) -> None:  
        """  
        Initialize the memory base class for either user or item memories.  
        新的memory base for两种类型的memory储存

        Args:  
            memory_type (str): Type of memory ("user" or "item").  
            llm: An instance of the language model (LLM) used for embeddings.  
        """  
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

    def retrieveMemory(self, query_scenario):  
        raise NotImplementedError("This method should be implemented by subclasses.")  

    def addMemory(self, user_data):  
        raise NotImplementedError("This method should be implemented by subclasses.")


        """  
        Add a user's data to the memory store.  

        Args:  
            user_data (dict): Dictionary with user-related fields.  
        """  
        # Create the metadata dynamically, setting missing fields to None  
        metadata = {  
            "user_id": user_data.get('user_id'),  
            "useful": user_data.get('useful', None),  
            "funny": user_data.get('funny', None),  
            "cool": user_data.get('cool', None),  
            "elite": user_data.get('elite', None),  
            "average_stars": user_data.get('average_stars', None),  
            "compliment_hot": user_data.get('compliment_hot', None),  
            "compliment_more": user_data.get('compliment_more', None),  
            "compliment_profile": user_data.get('compliment_profile', None),  
            "compliment_cute": user_data.get('compliment_cute', None),  
            "compliment_list": user_data.get('compliment_list', None),  
            "compliment_note": user_data.get('compliment_note', None),  
            "compliment_plain": user_data.get('compliment_plain', None),  
            "compliment_cool": user_data.get('compliment_cool', None),  
            "compliment_funny": user_data.get('compliment_funny', None),  
            "compliment_writer": user_data.get('compliment_writer', None),  
            "source": user_data.get('source', 'unknown')  # Default source if missing  
        }  

        # Create memory document  
        memory_doc = Document(  
            page_content=f"User: {user_data.get('user_id', 'unknown')}",  
            metadata=metadata  
        )  

        # Add memory document to the memory store  
        self.scenario_memory.add_documents([memory_doc])

class MemoryItem(MemoryBase):  
    def __init__(self, llm):  
        super().__init__(memory_type='item', llm=llm)  

    def retrieveMemory(self, query_scenario: str):
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

    #def addMemory(self, item_data: dict): 
    def addMemory(self, current_situation: str): 
        """  
        Add an item's data to the memory store. 存关于这个item的review

        Args:  
            item_data (dict): Dictionary with item-related fields.  
        """  
        """
        # 如果要批量存储？待修改
        memory_doc = Document(  
            page_content=f"Review: {item_data['text']}",  # 关于这个item的review是核心
            metadata={    
                "user": item_data.get('user_id', 'Unknown user'),  
                "text": item_data.get('text', 'No text available'),  # Store list of reviews  
                "stars": item_data.get('stars', None),  
                "useful": item_data.get('useful', None),  
                "funny": item_data.get('funny', None),  
                "cool": item_data.get('cool', None),  
                "source": item_data.get('source', 'unknown'),  # Default source if not provided   
            }  
        )  
        # Add to memory store  
        self.scenario_memory.add_documents([memory_doc])
        """
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
        
    def __call__(self, task_description: str):
        """Override the parent class's __call__ method"""
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

        
    def workflow(self):
        """
        Simulate user behavior
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        try:
            plan = self.planning(task_description=self.task) # 先规划任务，格式化返回

            for sub_task in plan:
                if 'user' in sub_task['description']:
                    user = self.interaction_tool.get_user(user_id=self.task['user_id']) # 对于sub_task，返回user和item_id
                elif 'business' in sub_task['description']:
                    business = str(self.interaction_tool.get_item(item_id=self.task['item_id'])) # 也可以操作
            reviews_item = self.interaction_tool.get_reviews(item_id=self.task['item_id']) # 根据item_id返回对应的reviews（多个）
            if not reviews_item:  # 检查是否有评论  
                print(f"No reviews found for item_id: {self.task['item_id']}")  
                return
            def analyze_user_style(metadata: dict) -> dict:  
                """  
                Analyze the user's style and tendencies based on their metadata.  

                Args:  
                    metadata (dict): Metadata containing user-related metrics.  

                Returns:  
                    dict: Analysis results for different aspects of the user.  
                """  
                # Rating tendency analysis  
                average_stars = metadata.get("average_stars", 0)  
                if average_stars >= 4.0:  
                    rating_tendency = f"This user tends to give high ratings (average stars: {average_stars:.1f}), indicating a more generous or satisfied personality."  
                elif average_stars < 3.0:  
                    rating_tendency = f"This user tends to give low ratings (average stars: {average_stars:.1f}), suggesting a critical or selective personality."  
                else:  
                    rating_tendency = f"This user gives balanced ratings (average stars: {average_stars:.1f}), reflecting a moderate and fair judgment style."  

                # Interaction style analysis  
                funny = metadata.get("funny", 0)  
                compliment_funny = metadata.get("compliment_funny", 0)  
                compliment_hot = metadata.get("compliment_hot", 0)  
                compliment_plain = metadata.get("compliment_plain", 0)  
                if funny > 10 or compliment_funny > 10:  
                    interaction_style = f"This user often displays a humorous tone in their reviews (funny: {funny}, compliment_funny: {compliment_funny})."  
                elif compliment_hot > 10:  
                    interaction_style = f"This user writes emotional and engaging reviews (compliment_hot: {compliment_hot})."  
                elif compliment_plain > 10:  
                    interaction_style = f"This user prefers a plain and straightforward tone in reviews (compliment_plain: {compliment_plain})."  
                else:  
                    interaction_style = "This user's interaction style is neutral or undefined."  

                # Content complexity analysis  
                elite = metadata.get("elite", False)  
                compliment_writer = metadata.get("compliment_writer", 0)  
                compliment_note = metadata.get("compliment_note", 0)  
                if elite or compliment_writer > 10 or compliment_note > 10:  
                    content_complexity = f"This user writes with a high degree of complexity and elegance (elite: {elite}, compliment_writer: {compliment_writer}, compliment_note: {compliment_note})."  
                else:  
                    content_complexity = "This user keeps reviews simple and easy to read."  

                # Activity level analysis  
                useful = metadata.get("useful", 0)  
                fans = metadata.get("fans", 0)  
                if useful > 50 or fans > 10:  
                    activity_level = f"This user is highly active and influential (useful: {useful}, fans: {fans})."  
                else:  
                    activity_level = "This user has a low to moderate activity level."  

                # Return all analyses as a dictionary  
                return {  
                    "rating_tendency": rating_tendency,  
                    "interaction_style": interaction_style,  
                    "content_complexity": content_complexity,  
                    "activity_level": activity_level,  
                }  
            def processDataUser(user: dict):  
                """  
                Retrieve memory for a specific user and analyze their behavior to generate a report.  

                Args:  
                    user: User information in dict.  

                Returns:  
                    str: A formatted string report summarizing the user's style and tendencies.  
                """   

                # Step 1: Analyze user style based on metadata  
                analysis = analyze_user_style(user)  
                #print(analysis)  

                # Step 2: Combine the analysis into a single prompt for the LLM  
                prompt = (f"Generate a detailed report analyzing the following user's review style and tendencies based on the provided analysis:\n\n"  
                        f"Rating Tendency:\n{analysis['rating_tendency']}\n\n"  
                        f"Interaction Style:\n{analysis['interaction_style']}\n\n"  
                        f"Content Complexity:\n{analysis['content_complexity']}\n\n"  
                        f"Activity Level:\n{analysis['activity_level']}\n\n"  
                        f"Summarize and explain the user's review style in a professional and engaging manner.")  

                # Step 3: Call LLM once to generate the full report  
                reasoning_result = self.llm(  
                    messages=[{"role": "user", "content": prompt}],  
                    temperature=0.5,  
                    max_tokens=2000  
                )  

                # Step 4: Return the generated report  
                return reasoning_result   

            # 对该item建立review的memory，感觉还有优化空间
            
            for review in reviews_item:
                #review_text = review['text'] # 用tool从review.json里取出来
                self.memory_item.addMemory(review) # 关于这个item的review
            #print(len(reviews_item))
            # 从item memory里提取similar reviews
            reviews_user = self.interaction_tool.get_reviews(user_id=self.task['user_id']) # 通过user_id来获取这个人之前做的一些review
            review_similar = self.memory_item.retrieveMemory(f'{reviews_user[0]["text"]}') # 基于这个人的review从memory里取出相关的对item的review
            
            # 分两个prompt，因为yelp的user有更多指标
            if user['source'] == 'yelp':
                # 从user meomory里提取user风格特征
                user_style = processDataUser(user) # 返回一段文本
                task_description = f'''
                You are a real human user on Yelp, a platform for crowd-sourced business reviews. 
                Here is your Yelp profile and review history style: {user_style}

                You need to write a review for this business: {business}

                Others have reviewed this business before: {review_similar}

                Please analyze the following aspects carefully:
                1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
                2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
                3. Consider how other users might engage with your review in terms of:
                - Useful: How informative and helpful is your review?
                - Funny: Does your review have any humorous or entertaining elements?
                - Cool: Is your review particularly insightful or praiseworthy?

                Requirements:
                - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
                - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
                - If the business fails significantly in key areas, consider giving a 1-star rating
                - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
                - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
                - Maintain consistency with your historical review style and rating patterns
                - Focus on specific details about the business rather than generic comments
                - Be generous with ratings when businesses deliver quality service and products
                - Be critical when businesses fail to meet basic standards

                Format your response exactly as follows:
                stars: [your rating]
                review: [your review]
                '''
            else: # for amazon and goodreads
                task_description = f'''
                You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

                You need to write a review for this business: {business}

                Others have reviewed this business before: {review_similar}

                Please analyze the following aspects carefully:
                1. Based on your user profile and review style, what rating would you give this business? Remember that many users give 5-star ratings for excellent experiences that exceed expectations, and 1-star ratings for very poor experiences that fail to meet basic standards.
                2. Given the business details and your past experiences, what specific aspects would you comment on? Focus on the positive aspects that make this business stand out or negative aspects that severely impact the experience.
                3. Consider how other users might engage with your review in terms of:
                - Useful: How informative and helpful is your review?
                - Funny: Does your review have any humorous or entertaining elements?
                - Cool: Is your review particularly insightful or praiseworthy?

                Requirements:
                - Star rating must be one of: 1.0, 2.0, 3.0, 4.0, 5.0
                - If the business meets or exceeds expectations in key areas, consider giving a 5-star rating
                - If the business fails significantly in key areas, consider giving a 1-star rating
                - Review text should be 2-4 sentences, focusing on your personal experience and emotional response
                - Useful/funny/cool counts should be non-negative integers that reflect likely user engagement
                - Maintain consistency with your historical review style and rating patterns
                - Focus on specific details about the business rather than generic comments
                - Be generous with ratings when businesses deliver quality service and products
                - Be critical when businesses fail to meet basic standards

                Format your response exactly as follows:
                stars: [your rating]
                review: [your review]
                '''
            #print(task_description)
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

if __name__ == "__main__":
    # Set the data
    # 获取脚本所在目录的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 获取项目根目录路径  
    root_dir = os.path.dirname(script_dir)  

    # 数据目录路径
    task_set = "yelp" # "goodreads" or "yelp"
    data_dir = os.path.join(root_dir, "data", "processed")  
    simulator = Simulator(data_dir=data_dir, device="gpu", cache=True)  
    
    task_dir = os.path.join(script_dir, "track1", task_set, "tasks")  
    groundtruth_dir = os.path.join(script_dir, "track1", task_set, "groundtruth")  
    simulator.set_task_and_groundtruth(task_dir=task_dir, groundtruth_dir=groundtruth_dir)  

    # Set the agent and LLM
    simulator.set_agent(MySimulationAgent)
    simulator.set_llm(InfinigenceLLM(api_key="sk-damtshfyvhcd7xmg"))

    # Run the simulation
    # If you don't set the number of tasks, the simulator will run all tasks.
    outputs = simulator.run_simulation(number_of_tasks=100, enable_threading=True, max_workers=10)
    
    # Evaluate the agent
    evaluation_results = simulator.evaluate()       
    with open(f'./evaluation_results_track1_{task_set}.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)

    # Get evaluation history
    evaluation_history = simulator.get_evaluation_history()
