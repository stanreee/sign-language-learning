import { NavLink } from 'react-router-dom';
import SectionGoal from '../components/SectionGoal';
import SectionInfo from '../components/SectionInfo';
import SectionResources from '../components/SectionResources';
import asl from '../images/asl.png';

console.log(asl);

const Home = () => {
    return(
    <div className="App">
      <header className="App-header">
        <img src={asl} alt="asl" />
        <h1>Welcome to ASLingo!</h1>
        <a>An Application to Learn ASL, American Sign Language</a>
        
        <div className = 'box'>
          <NavLink 
            to="/exercises"
            style={({ isActive, isPending }) => {
              return {
                fontWeight: isActive ? "bold" : "bold",
                color: isPending ? "red" : '#003459',
              };
            }}
          >
            Get Started
          </NavLink>
        </div>
      </header>

      <SectionGoal />
      <SectionInfo />
      <SectionResources />
    </div>
    );
}

export default Home;
