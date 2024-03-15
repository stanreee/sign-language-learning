import { NavLink } from 'react-router-dom';
import info from "../data/fyInfo.json"
import asl from '../images/asl.png';
import { useEffect } from 'react';

console.log(asl);

const Home = () => {

    //DATA CALL
    const goalsTitle = info.goalHeader; 
    const goalsDesc = info.goalDesc.join(" "); 
    const backgroundTitle = info.backgroundInfoHeader;
    const backgroundDesc = info.backgroundInfoDesc.join(" ");
    const resInfoTitle = info.resourcesInfoHeader;
    const resInfoDesc = info.resourcesInfoDesc.join(" ");

    return(
    <div className="App">
      <header className="App-header">
        <img src={asl} alt="asl" />
        <h1>Welcome to ASLingo!</h1>
        <p>An Application to Help People Learn American Sign Language (ASL)</p>
        
        <h3> <br/> </h3>
        
        <div className = 'box'>
          <NavLink reloadDocument to={'/learn'}
            style={({ isActive, isPending }) => {
              return {
                fontWeight: isActive ? "bold" : "bold",
                color: isPending ? "red" : '#003459',
              };
            }}
          >
            Start Learning
          </NavLink>
        </div>
      </header>

      <div className = 'Section-container'>
        <div className = "Section">
            <h1> {goalsTitle} </h1>
            <p> {goalsDesc} </p>
        </div>

        <div className = "Section">
            <h1> {backgroundTitle} </h1>
            <p> {backgroundDesc} </p>
        </div>

        <div className = "Section">
            <h1> {resInfoTitle} </h1>
            <p> {resInfoDesc} </p>
            <NavLink reloadDocument to={'/learn'}
						style={({ isActive, isPending }) => {
							return {
							fontWeight: isActive ? "bold" : "bold",
							color: isPending ? "red" : '#fff',
							};
						}} >
						Click here to Learn more!
			      </NavLink>
        </div>
      </div>
    </div>
    );
}

export default Home;
