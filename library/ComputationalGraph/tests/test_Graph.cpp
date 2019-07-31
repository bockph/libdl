//
// Created by phili on 15.05.2019.
//
#include <Graph.hpp>
#include <catch2/catch.hpp>
#include <Placeholder.hpp>
#include <OperationsFactory.hpp>


TEST_CASE("Graph Forwardpass ", "[operation]") {

    SECTION("Graph should not fail when adding an operation whos input is already set", "") {
        auto graph = std::make_shared<Graph>();
        Matrix X;
        auto realInput= std::make_shared<Placeholder>(X,1);
        graph->setInput(realInput);
        //This should not fail
        //This should fail as its Input is not the last operation
        bool test =true;
        try
        {
            //this should not fail
            OperationsFactory::createSigmoidOp(graph,realInput);
        }
        catch (const std::runtime_error& error)
        {
            test =false;
            // your error handling code here
        }
        REQUIRE(test ==true);
    }
    SECTION("Graph should fail when adding an operation before setting Input", "") {
        auto graph = std::make_shared<Graph>();
        Matrix X;
        auto fakeInput= std::make_shared<Placeholder>(X,1);
        bool test =false;
        try
        {
            OperationsFactory::createSigmoidOp(graph,fakeInput);

        }
        catch (const std::runtime_error& error)
        {
            test =true;
            // your error handling code here
        }
        REQUIRE(test ==true);
    }
    SECTION("Graph should fail when adding an operation whos input is not directly", "") {
        auto graph = std::make_shared<Graph>();
        Matrix X;
        auto realInput= std::make_shared<Placeholder>(X,1);
        graph->setInput(realInput);
        //This should not fail
        OperationsFactory::createSigmoidOp(graph,realInput);
        //This should fail as its Input is not the last operation
        bool test =false;
        try
        {
            OperationsFactory::createSigmoidOp(graph,realInput);

        }
        catch (const std::runtime_error& error)
        {
            test =true;
            // your error handling code here
        }
        REQUIRE(test ==true);
    }

}