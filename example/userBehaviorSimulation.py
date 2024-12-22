from websocietysimulator import Simulator
from websocietysimulator.agent import SimulationAgent
import json 
from websocietysimulator.llm import DeepseekLLM
from websocietysimulator.agent.modules.planning_modules import PlanningBase 
from websocietysimulator.agent.modules.reasoning_modules import ReasoningBase
import os

api_key = os.getenv('DEEPSEEK_API_KEY')
class PlanningBaseline(PlanningBase):
    """继承自 PlanningBase 的规划模块"""
    
    def __init__(self, llm):
        """初始化规划模块"""
        super().__init__(llm=llm)
    
    def __call__(self, task_description):
        """重写父类的 __call__ 方法"""
        self.plan = [
            {
                'description': 'First I need to find user information',
                'reasoning instruction': 'None', 
                'tool use instruction': {task_description['user_id']}
            },
            {
                'description': 'Next, I need to find business information',
                'reasoning instruction': 'None',
                'tool use instruction': {task_description['business_id']}
            }
        ]
        return self.plan


class ReasoningBaseline(ReasoningBase):
    """继承自 ReasoningBase 的推理模块"""
    
    def __init__(self, profile_type_prompt, llm):
        """初始化推理模块"""
        super().__init__(profile_type_prompt=profile_type_prompt, memory=None, llm=llm)
        
    def __call__(self, task_description: str):
        """重写父类的 __call__ 方法"""
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
    
    def __init__(self):
        """初始化 MySimulationAgent"""
        super().__init__()
        self.llm = DeepseekLLM(api_key=api_key)
        self.planning = PlanningBaseline(llm=self.llm)
        self.reasoning = ReasoningBaseline(profile_type_prompt='', llm=self.llm)

    def forward(self):
        """
        模拟用户行为
        Returns:
            tuple: (star (float), useful (float), funny (float), cool (float), review_text (str))
        """
        plan = self.planning(task_description=self.scenario)

        for sub_task in plan:
            if 'user' in sub_task['description']:
                user = str(self.interaction_tool.get_user(user_id=self.scenario['user_id']))
            elif 'business' in sub_task['description']:
                business = str(self.interaction_tool.get_business(business_id=self.scenario['business_id']))

        task_description = f'''
        You are a real human user on Yelp, a platform for crowd-sourced business reviews. Here is your Yelp profile and review history: {user}

        You need to write a review for this business: {business}

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
        star: [your rating]
        useful: [count]
        funny: [count] 
        cool: [count]
        review_text: [your review]
        '''
        result = self.reasoning(task_description)
        
        try:
            star_line = [line for line in result.split('\n') if 'star:' in line][0]
            review_line = [line for line in result.split('\n') if 'review_text:' in line][0]
            useful_line = [line for line in result.split('\n') if 'useful:' in line][0]
            funny_line = [line for line in result.split('\n') if 'funny:' in line][0]
            cool_line = [line for line in result.split('\n') if 'cool:' in line][0]
        except:
            print('Error:', result)

        star = float(star_line.split(':')[1].strip())
        useful = float(useful_line.split(':')[1].strip())
        funny = float(funny_line.split(':')[1].strip())
        cool = float(cool_line.split(':')[1].strip())
        review_text = review_line.split(':')[1].strip()

        if len(review_text) > 512:
            review_text = review_text[:512]
            
        return star, useful, funny, cool, review_text


if __name__ == "__main__":
    simulator = Simulator(data_dir="xxx")
    simulator.set_task_and_groundtruth(task_dir="xxx", groundtruth_dir="xxx")
    simulator.set_agent(MySimulationAgent)
    outputs = simulator.run_simulation()

    evaluation_results = simulator.evaluate()       
    with open('./evaluation_results.json', 'w') as f:
        json.dump(evaluation_results, f, indent=4)