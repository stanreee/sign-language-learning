import { NavLink } from 'react-router-dom';
import asl from '../images/asl.png';
// import { useNavigate } from "react-router-dom";

console.log(asl);

const Account = () => {

  // const { loggedIn, email } = props
  // const navigate = useNavigate();
  
  // const onButtonClick = () => {
  //   navigate("/login")
  //   if (loggedIn) {
  //       localStorage.removeItem("user")
  //       props.setLoggedIn(false)
  //   } else {
  //       navigate("/login")
  //   }
  // }


    return(

      // <div className="mainContainer">
      //   <div className={"titleContainer"}>
      //       <div>Welcome to ASLingo</div>
      //   </div>
      //   <div>
      //       Login to save your learning progress!
      //   </div>
      //   <div className={"buttonContainer"}>
      //       <input
      //           className={"inputButton"}
      //           type="button"
      //           onClick={onButtonClick}
      //           value={loggedIn ? "Log out" : "Log in"} />
      //       {(loggedIn ? <div>
      //           Your email address is {email}
      //       </div> : <div/>)}
      //   </div>
      // </div>


    <div className="App">
      <header className = "App-header">
        <img src={asl} alt="asl" />
        <h1>Welcome to ASLingo!</h1>
        <div>
          Login to save your learning progress!
        </div>

        <div className = 'box'>
          <NavLink 
            to="/login"
            style={({ isActive, isPending }) => {
              return {
                fontWeight: isActive ? "bold" : "bold",
                color: isPending ? "red" : '#003459',
              };
            }}
          >
            Login
          </NavLink>
        </div>
      </header>
    </div>
    );
}

export default Account;