import SectionGoal from '../components/SectionGoal';
import SectionInfo from '../components/SectionInfo';
import SectionResources from '../components/SectionResources';

const Home = () => {
    return(
    <div className="App">
      <header className="App-header">
        <h1>Welcome to ASLingo!</h1>
        <a>An Application to Learn ASL, American Sign Language</a>
      </header>

      <SectionGoal />
      <SectionInfo />
      <SectionResources />
    </div>
    );
}

export default Home;